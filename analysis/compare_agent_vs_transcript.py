#!/usr/bin/env python3
"""Create a case-level audit of agent entries versus transcript extraction."""

import csv
import random
import unicodedata
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = ANALYSIS_DIR.parent
SOURCE = ROOT_DIR / "transcribe cases .csv"
AI = ANALYSIS_DIR / "gpt_results/ollama_transcript_analysis_en.csv"
OUTPUT = ANALYSIS_DIR / "agent_vs_transcript_comparison.csv"
SUMMARY = ANALYSIS_DIR / "agent_vs_transcript_summary.csv"
REVIEW = ANALYSIS_DIR / "agent_vs_transcript_review_sample.csv"


def normalize(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    return " ".join("".join(c for c in value if not unicodedata.combining(c)).split())


GENERIC_REASONS = {"", "nao informado", "nao informou", "nao fui informado", "n/a", "nao informada"}
NO_SIGNAL = {"", "none_stated", "unclear"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    source_rows = read_csv(SOURCE)
    ai_rows = read_csv(AI)
    source = {row["task_id"]: row for row in source_rows}
    ai = {row["task_id"]: row for row in ai_rows}
    if len(source) != len(source_rows) or len(ai) != len(ai_rows) or set(source) != set(ai):
        raise ValueError("task_id must be unique and present in both files")

    output_rows = []
    for task_id, s in source.items():
        a = ai[task_id]
        rtp = s["call_result_label"].startswith("RTP")
        agent_reason_generic = normalize(s["rtp_reason"]) in GENERIC_REASONS
        hardship = a["hardship_category"] not in NO_SIGNAL
        other = a["other_nonpayment_reason"] not in NO_SIGNAL
        actionable = hardship or other
        evidence_exact = bool(a["evidence"]) and normalize(a["evidence"]) in normalize(s["full_conversation"])
        candidate = rtp and agent_reason_generic and actionable
        supported_candidate = candidate and evidence_exact and a["confidence"] in {"medium", "high"}
        output_rows.append({
            "task_id": task_id,
            "user_id": s["user_id"],
            "call_info_id": s["call_info_id"],
            "case_id": s["case_id"],
            "agent_call_result": s["call_result_label"],
            "agent_rtp_reason": s["rtp_reason"],
            "agent_reason_status": ("not_applicable_ptp" if not rtp else "blank_or_generic" if agent_reason_generic else "specific"),
            "agent_ptp_amount": s["ptp_amount"],
            "agent_ptp_due_date": s["ptp_due_date"],
            "ai_hardship_category": a["hardship_category"],
            "ai_hardship_description": a["hardship_description"],
            "ai_other_nonpayment_reason": a["other_nonpayment_reason"],
            "ai_agent_response_type": a["agent_response_type"],
            "ai_conversation_result": a["conversation_result"],
            "ai_promise_status": a["promise_status"],
            "ai_amount_promised": a["amount_promised"],
            "ai_payment_date_spoken": a["payment_date_spoken"],
            "ai_confidence": a["confidence"],
            "ai_evidence": a["evidence"],
            "ai_evidence_exact_in_transcript": int(evidence_exact),
            "ai_actionable_reason": int(actionable),
            "candidate_added_reason_rtp": int(candidate),
            "review_candidate_added_reason_rtp": int(supported_candidate),
            "human_review": "",
        })

    fields = list(output_rows[0])
    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    rtp_rows = [r for r in output_rows if r["agent_call_result"].startswith("RTP")]
    ptp_rows = [r for r in output_rows if r["agent_call_result"].startswith("PTP")]
    metrics = [
        ("all_calls", len(output_rows)),
        ("rtp_calls", len(rtp_rows)),
        ("ptp_calls", len(ptp_rows)),
        ("rtp_agent_reason_blank_or_generic", sum(r["agent_reason_status"] == "blank_or_generic" for r in rtp_rows)),
        ("rtp_ai_actionable_reason", sum(int(r["ai_actionable_reason"]) for r in rtp_rows)),
        ("rtp_candidate_added_reason", sum(int(r["candidate_added_reason_rtp"]) for r in rtp_rows)),
        ("rtp_review_candidates_with_matching_evidence", sum(int(r["review_candidate_added_reason_rtp"]) for r in rtp_rows)),
        ("ptp_ai_actionable_reason", sum(int(r["ai_actionable_reason"]) for r in ptp_rows)),
        ("ai_evidence_not_exact", sum(not int(r["ai_evidence_exact_in_transcript"]) for r in output_rows)),
        ("ai_amount_with_nonconfirmed_status", sum(bool(r["ai_amount_promised"]) and r["ai_promise_status"] != "confirmed" for r in output_rows)),
    ]
    with SUMMARY.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "count"])
        writer.writerows(metrics)

    # Fixed-seed strata make review repeatable and avoid selecting only apparent successes.
    rng = random.Random(20261022)
    cohorts = {
        "RTP missing agent reason, AI adds detail": [r for r in output_rows if r["review_candidate_added_reason_rtp"]],
        "RTP agent says uncooperative": [r for r in output_rows if r["agent_rtp_reason"] == "Unwill - Not cooperative"],
        "PTP with AI reason": [r for r in output_rows if r["agent_call_result"].startswith("PTP") and r["ai_actionable_reason"]],
        "RTP no clear AI addition": [r for r in output_rows if r["agent_reason_status"] == "blank_or_generic" and not r["ai_actionable_reason"]],
    }
    review_rows = []
    seen = set()
    for cohort, candidates in cohorts.items():
        for r in rng.sample(candidates, min(10, len(candidates))):
            if r["task_id"] in seen:
                continue
            seen.add(r["task_id"])
            review_rows.append({
                "cohort": cohort,
                **r,
                "full_conversation": source[r["task_id"]]["full_conversation"],
                "human_reason_supported": "",
                "human_adds_useful_information": "",
                "human_notes": "",
            })
    with REVIEW.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"Wrote {len(output_rows)} cases to {OUTPUT}, {len(metrics)} metrics to {SUMMARY}, and {len(review_rows)} review cases to {REVIEW}")


if __name__ == "__main__":
    main()
