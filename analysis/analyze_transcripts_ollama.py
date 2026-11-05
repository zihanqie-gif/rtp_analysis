#!/usr/bin/env python3
"""Extract collection-call information from transcripts with a local Ollama model."""

import argparse
import csv
import json
import re
import sys
import time
import shutil
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent


INPUT_FIELDS = ("task_id", "user_id", "call_info_id", "full_conversation")
CATEGORIES = {
    "customer_sentiment": ["positive", "neutral", "negative", "mixed", "unclear"],
    "customer_tone": ["cooperative", "worried", "frustrated", "angry", "confused", "resigned", "guarded", "other", "unclear"],
    "hardship_category": ["waiting_for_income", "job_loss_or_reduced_income", "increased_expenses", "health_issue", "temporary_cash_shortage", "instalments_unaffordable", "other_financial_hardship", "none_stated", "unclear"],
    "other_nonpayment_reason": ["needs_restructuring", "disputes_balance", "already_paid", "awaiting_refund_or_product_resolution", "payment_or_app_difficulty", "disputes_account_or_contact", "unwilling_to_pay", "none_stated", "other", "unclear"],
    "agent_response_type": ["asks_about_circumstances", "offers_restructuring", "offers_discount", "requests_full_payment", "records_promise", "explains_consequences", "gives_payment_instructions", "refers_to_support", "requested_option_unavailable", "no_substantive_response", "other"],
    "conversation_result": ["clear_promise", "vague_intention", "conditional_willingness", "proposal_unaffordable", "negotiation_pending", "explicit_refusal", "already_paid_claim", "unresolved_dispute", "no_clear_outcome", "other"],
    "promise_status": ["confirmed", "conditional", "agent_proposed_only", "none", "unclear"],
}
FIELDS = [
    "task_id", "user_id", "call_info_id", "customer_sentiment", "customer_tone",
    "hardship_description", "hardship_category", "other_nonpayment_reason",
    "agent_response_type", "conversation_result", "promise_status",
    "amount_promised", "payment_date_spoken", "payment_date_iso",
    "products_talked_about", "evidence", "confidence", "review_issues", "error",
]
SCHEMA = {
    "type": "object",
    "properties": {
        **{name: {"type": "string", "enum": values} for name, values in CATEGORIES.items()},
        "hardship_description": {"type": "string"},
        "amount_promised": {"type": ["number", "null"]},
        "payment_date_spoken": {"type": "string"},
        "payment_date_iso": {"type": "string"},
        "products_talked_about": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [*CATEGORIES, "hardship_description", "amount_promised", "payment_date_spoken",
                 "payment_date_iso", "products_talked_about", "evidence", "confidence"],
    "additionalProperties": False,
}
PROMPT = """Analyze this Brazilian Portuguese debt-collection call. Return JSON only.
Use ONLY the transcript below. The instructions contain no facts about this call. Distinguish Customer from Agent and read through the end of the call.
Hardship: classify only a financial difficulty stated by the customer. An unaffordable instalment is hardship; a requested repayment arrangement, app problem, refund, or balance dispute is an other_nonpayment_reason. Do not infer salary, job loss, inability to afford, or unwillingness from a negotiation request.
Outcome: use the customer's final position. A proposed payment arrangement is negotiation_pending unless accepted. A customer who wants to pay under different terms is not unwilling_to_pay.
Promise: confirmed only when the customer says they will pay or clearly accepts a specific payment. A condition or proposal is not confirmed. amount_promised must be null unless the customer states or explicitly accepts that exact amount. Never use total debt or an agent's unaccepted quote as a promise.
Dates and products: payment_date_spoken must be an exact phrase from a Customer turn, or a date from an Agent turn explicitly accepted by the customer. Otherwise use an empty string. Use an empty payment_date_iso unless a full date including year is stated. Include only product names that are visibly present in the transcript; otherwise use an empty list.
Evidence: copy one short, continuous phrase exactly from a single transcript turn. Never combine phrases. Exclude names, phone numbers, and account identifiers. Use low confidence when speech recognition is unclear.

Transcript:
"""


def ask_ollama(url, model, transcript, timeout, context, think):
    body = json.dumps({
        "model": model,
        "prompt": PROMPT + transcript,
        "stream": False,
        "think": think,
        "format": SCHEMA,
        "options": {"temperature": 0, "num_ctx": context},
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url.rstrip("/") + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    answer = json.loads(result["response"])
    for field, allowed in CATEGORIES.items():
        if answer.get(field) not in allowed:
            raise ValueError(f"invalid {field}: {answer.get(field)!r}")
    if not isinstance(answer.get("products_talked_about"), list):
        raise ValueError("products_talked_about must be a list")
    if not isinstance(answer.get("amount_promised"), (int, float, type(None))):
        raise ValueError("amount_promised must be a number or null")
    return answer


def normalized(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    return " ".join("".join(c for c in value if not unicodedata.combining(c)).split())


def check_grounding(answer, transcript):
    issues = []
    turns = transcript.splitlines()
    customer = "\n".join(line.removeprefix("Customer: ") for line in turns if line.startswith("Customer: "))
    all_text = normalized(transcript)
    customer_text = normalized(customer)

    evidence = normalized(answer.get("evidence", ""))
    if evidence and not any(evidence in normalized(line) for line in turns):
        issues.append("evidence_not_verbatim")
        answer["evidence"] = ""

    spoken_date = normalized(answer.get("payment_date_spoken", ""))
    if spoken_date in {"nao especificado", "not specified", "none", "n/a"}:
        answer["payment_date_spoken"] = ""
    elif spoken_date and spoken_date not in customer_text:
        issues.append("payment_date_not_in_customer_words")
        answer["payment_date_spoken"] = ""

    kept_products = []
    for product in answer.get("products_talked_about", []):
        if not isinstance(product, str):
            issues.append("invalid_product")
        elif normalized(product).replace(" ", "") in all_text.replace(" ", ""):
            kept_products.append(product)
        else:
            issues.append("product_not_spoken")
    answer["products_talked_about"] = kept_products

    description = normalized(answer.get("hardship_description", ""))
    income_words = ("salary", "salario", "wage", "paycheck", "pay day", "renda", "receber", "adiantamento")
    transcript_income_words = ("salario", "salário", "renda", "receber", "adiantamento", "pagamento")
    if any(word in description for word in income_words) and not any(word in customer_text for word in transcript_income_words):
        issues.append("income_claim_not_grounded")
    if answer.get("promise_status") in {"none", "agent_proposed_only", "unclear"} and answer.get("amount_promised") is not None:
        issues.append("amount_without_confirmed_promise")
        answer["amount_promised"] = None
    if issues:
        answer["confidence"] = "low"
    return ";".join(sorted(set(issues)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ANALYSIS_DIR / "transcripts_for_ollama.csv")
    parser.add_argument("--output", type=Path, default=ANALYSIS_DIR / "ollama_transcript_analysis.csv")
    parser.add_argument("--model", required=True, help="Installed Ollama model name")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--limit", type=int, help="Process at most this many new rows")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--context", type=int, default=8192, help="Ollama context tokens (default: 8192)")
    parser.add_argument("--think", action="store_true", help="Enable model reasoning; slower but may improve classification")
    parser.add_argument("--retries", type=int, default=0)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must differ")

    with args.input.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not set(INPUT_FIELDS).issubset(reader.fieldnames or []):
            parser.error(f"input must contain {', '.join(INPUT_FIELDS)}")
        rows = list(reader)
    task_ids = [row["task_id"] for row in rows]
    if len(task_ids) != len(set(task_ids)):
        parser.error("task_id must be unique in the input")

    completed = set()
    if args.output.exists():
        with args.output.open(encoding="utf-8", newline="") as existing:
            prior = csv.DictReader(existing)
            if prior.fieldnames != FIELDS:
                parser.error("existing output has different columns; choose another --output")
            completed = {row["task_id"] for row in prior}

    new_file = not args.output.exists()
    processed = 0
    failures = 0
    pending = [row for row in rows if row["task_id"] not in completed]
    if args.limit is not None:
        pending = pending[:args.limit]
    total = len(pending)
    started = time.monotonic()
    with args.output.open("a", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
            destination.flush()
        for row in pending:
            if sys.stderr.isatty():
                print(f"\r\033[2KAnalyzing call {processed + 1}/{total} (task_id={row['task_id']}); waiting for Ollama...", end="", file=sys.stderr, flush=True)
            else:
                print(f"Analyzing call {processed + 1}/{total} (task_id={row['task_id']}); waiting for Ollama...", file=sys.stderr, flush=True)
            result = None
            error = ""
            for attempt in range(args.retries + 1):
                try:
                    result = ask_ollama(args.url, args.model, row["full_conversation"], args.timeout, args.context, args.think)
                    break
                except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < args.retries:
                        time.sleep(min(2 ** attempt, 8))
            output = {field: "" for field in FIELDS}
            output.update({field: row[field] for field in INPUT_FIELDS[:3]})
            if result is not None:
                output["review_issues"] = check_grounding(result, row["full_conversation"])
                output.update(result)
                output["products_talked_about"] = json.dumps(result["products_talked_about"], ensure_ascii=False)
            else:
                output["error"] = error
                failures += 1
            writer.writerow(output)
            destination.flush()
            processed += 1
            if sys.stderr.isatty():
                width = min(40, max(10, shutil.get_terminal_size().columns - 66))
                filled = round(width * processed / total)
                elapsed = int(time.monotonic() - started)
                bar = "#" * filled + "-" * (width - filled)
                print(f"\r\033[2K[{bar}] {processed}/{total} | errors {failures} | {elapsed}s", end="", file=sys.stderr, flush=True)
            else:
                print(f"{processed}/{total}: task_id={row['task_id']} {'OK' if result is not None else 'ERROR ' + error}", file=sys.stderr)
    if sys.stderr.isatty() and processed:
        print(file=sys.stderr)
    print(f"Processed {processed} new calls ({failures} errors). Output: {args.output}")


if __name__ == "__main__":
    main()
