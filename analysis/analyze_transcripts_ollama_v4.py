import argparse
import csv
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

from analyze_transcripts_ollama import CATEGORIES

ANALYSIS_DIR = Path(__file__).resolve().parent


QUOTE_FIELDS = {
    "hardship_quote": "Customer's exact words describing a financial hardship; empty if none",
    "reason_quote": "Customer's exact words explaining nonpayment or requested terms; empty if none",
    "promise_quote": "Customer's exact words explicitly promising a payment; empty if none",
    "outcome_quote": "Customer's exact words indicating their final position; empty if none",
    "agent_quote": "Agent's exact words showing the main response/action; empty if none",
    "product_quote": "One exact line mentioning the product name; empty if none",
}
QUOTE_SCHEMA = {
    "type": "object",
    "properties": {key: {"type": "string"} for key in QUOTE_FIELDS},
    "required": list(QUOTE_FIELDS),
    "additionalProperties": False,
}
LABEL_FIELDS = (
    "customer_sentiment", "customer_tone", "hardship_category", "other_nonpayment_reason",
    "agent_response_type", "conversation_result", "promise_status",
)
LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        **{key: {"type": "string", "enum": CATEGORIES[key]} for key in LABEL_FIELDS},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [*LABEL_FIELDS, "confidence"],
    "additionalProperties": False,
}
OUTPUT_FIELDS = [
    "task_id", "user_id", "call_info_id", *LABEL_FIELDS,
    "hardship_description", "amount_promised", "payment_date_spoken", "payment_date_iso",
    "products_talked_about", *QUOTE_FIELDS, "confidence", "review_issues", "error",
]


def norm(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    return " ".join("".join(c for c in value if not unicodedata.combining(c)).split())


def generate(url, model, prompt, schema, timeout, context, think):
    payload = json.dumps({
        "model": model, "prompt": prompt, "format": schema, "stream": False,
        "think": think, "options": {"temperature": 0, "num_ctx": context},
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(json.load(response)["response"])


def verified_quotes(raw, transcript):
    lines = transcript.splitlines()
    quotes = {}
    issues = []
    for key in QUOTE_FIELDS:
        quote = raw.get(key, "")
        if not isinstance(quote, str):
            quote = ""
        quote = quote.strip().strip('"')
        role = "Agent:" if key == "agent_quote" else "Customer:"
        candidates = lines if key == "product_quote" else [line for line in lines if line.startswith(role)]
        if quote and any(norm(quote) in norm(line) for line in candidates):
            quotes[key] = quote
        else:
            quotes[key] = ""
            if quote:
                issues.append(key + "_not_verbatim")
    return quotes, issues


def amount_from_customer_promise(quote):
    if not re.search(r"\b(pag|quit|depos|transfer|pix|fa[cç]o\s+o\s+pagamento)", norm(quote)):
        return None
    matches = re.findall(r"(?:r\$?\s*)?(\d{1,3}(?:\.\d{3})*,\d{2})\b", quote, re.I)
    if len(matches) != 1:
        return None
    return float(matches[0].replace(".", "").replace(",", "."))


def date_from_customer_promise(quote):
    if not re.search(r"\b(pag|quit|depos|transfer|pix|fa[cç]o\s+o\s+pagamento)", norm(quote)):
        return ""
    for pattern in [r"\b(hoje|amanh[ãa]|depois de amanh[ãa])\b",
                    r"\b(dia\s+\d{1,2}(?:\s+de\s+\w+)?)\b",
                    r"\b(segunda|ter[cç]a|quarta|quinta|sexta|s[áa]bado|domingo)(?:-feira)?\b"]:
        match = re.search(pattern, quote, re.I)
        if match:
            return match.group(0)
    return ""


def products_from_quote(quote):
    text = norm(quote).replace(" ", "")
    products = []
    for name, pattern in [("SParcelado", "sparcelado"), ("SCrédito", "scredito"),
                          ("ShopeePay", "shopeepay"), ("Empréstimo pessoal", "emprestimopessoal")]:
        if pattern in text:
            products.append(name)
    return products


def classify(url, model, quotes, customer_turns, timeout, context, think):
    prompt = (
        "Classify this Brazilian Portuguese collection call using ONLY the verified quotes below. "
        "Missing quotes mean no evidence. Do not invent hardship, payment amount, date, or product. "
        "A customer asking to negotiate or proposing an arrangement is not unwilling_to_pay. "
        "A conditional proposal is not a confirmed promise. An unaffordable instalment is instalments_unaffordable. "
        "Use the customer's final position for conversation_result. Return JSON only.\n\n"
        + json.dumps({"verified_quotes": quotes, "customer_turns": customer_turns}, ensure_ascii=False)
    )
    labels = generate(url, model, prompt, LABEL_SCHEMA, timeout, context, think)
    for key in LABEL_FIELDS:
        if labels.get(key) not in CATEGORIES[key]:
            raise ValueError(f"invalid {key}: {labels.get(key)!r}")
    return labels


def analyze(url, model, transcript, timeout, context, think):
    first_prompt = (
        "Extract short, exact, continuous quotations from this Brazilian Portuguese call. "
        "Return JSON only. Do not paraphrase or combine turns. Copy each quote from ONE transcript line. "
        "Use empty string if there is no clear quote. Customer and Agent roles are explicit. "
        "For promise_quote, quote only a Customer line explicitly saying they will pay, not a question, "
        "agent proposal, or conditional offer. For hardship_quote, quote only a Customer line stating "
        "financial difficulty. For outcome_quote, prefer the customer's final position.\n"
        + json.dumps(QUOTE_FIELDS, ensure_ascii=False) + "\n\nTranscript:\n" + transcript
    )
    raw = generate(url, model, first_prompt, QUOTE_SCHEMA, timeout, context, think)
    quotes, issues = verified_quotes(raw, transcript)
    customer_turns = [line.removeprefix("Customer: ") for line in transcript.splitlines()
                      if line.startswith("Customer: ")]
    labels = classify(url, model, quotes, customer_turns, timeout, context, think)

    # Guardrails override unsupported model judgments, favoring blank over invented facts.
    if not quotes["hardship_quote"]:
        labels["hardship_category"] = "none_stated"
    hardship = norm(quotes["hardship_quote"])
    if re.search(r"(parcela|parcelamento|prestacao).{0,50}(alto|alta|caro|cara|pesad|nao consigo|nao posso)", hardship) or re.search(r"(alto|alta|caro|cara).{0,50}(parcela|parcelamento|prestacao)", hardship):
        labels["hardship_category"] = "instalments_unaffordable"
    reason = norm(quotes["reason_quote"] + " " + quotes["outcome_quote"])
    if re.search(r"(parcel|reparcel|negoci|acordo)", reason) and labels["other_nonpayment_reason"] == "unwilling_to_pay":
        labels["other_nonpayment_reason"] = "needs_restructuring"
        issues.append("unwilling_label_overridden")
    if not quotes["promise_quote"]:
        labels["promise_status"] = "none"
    amount = amount_from_customer_promise(quotes["promise_quote"])
    date = date_from_customer_promise(quotes["promise_quote"])
    if labels["promise_status"] != "confirmed":
        amount = None
        date = ""
    if issues:
        labels["confidence"] = "low"
    return {**labels, **quotes, "hardship_description": quotes["hardship_quote"],
            "amount_promised": amount, "payment_date_spoken": date, "payment_date_iso": "",
            "products_talked_about": json.dumps(products_from_quote(quotes["product_quote"]), ensure_ascii=False),
            "review_issues": ";".join(sorted(set(issues)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ANALYSIS_DIR / "transcripts_for_ollama.csv")
    parser.add_argument("--output", type=Path, default=ANALYSIS_DIR / "ollama_analysis_evidence_first.csv")
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must differ")
    with args.input.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"task_id", "user_id", "call_info_id", "full_conversation"}
    if rows and not required.issubset(rows[0]):
        parser.error("input lacks required columns")
    if len({r["task_id"] for r in rows}) != len(rows):
        parser.error("task_id must be unique")
    completed = set()
    if args.output.exists():
        with args.output.open(encoding="utf-8", newline="") as f:
            prior = csv.DictReader(f)
            if prior.fieldnames != OUTPUT_FIELDS:
                parser.error("output has different columns; choose a fresh --output")
            completed = {r["task_id"] for r in prior}
    pending = [r for r in rows if r["task_id"] not in completed]
    if args.limit is not None:
        pending = pending[:args.limit]
    new = not args.output.exists()
    errors = 0
    with args.output.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if new:
            writer.writeheader()
            f.flush()
        for i, row in enumerate(pending, 1):
            print(f"Analyzing {i}/{len(pending)} task_id={row['task_id']} (two model steps)...", file=sys.stderr, flush=True)
            record = {key: "" for key in OUTPUT_FIELDS}
            record.update({key: row[key] for key in ("task_id", "user_id", "call_info_id")})
            try:
                record.update(analyze(args.url, args.model, row["full_conversation"], args.timeout, args.context, args.think))
            except Exception as exc:
                errors += 1
                record["error"] = f"{type(exc).__name__}: {exc}"
            writer.writerow(record)
            f.flush()
            print(f"[{i}/{len(pending)}] {'OK' if not record['error'] else record['error']}", file=sys.stderr, flush=True)
    print(f"Processed {len(pending)} calls ({errors} errors): {args.output}")


if __name__ == "__main__":
    main()
