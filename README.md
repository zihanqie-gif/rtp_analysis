# RTP call-reason analysis

This project explores whether call transcripts contain a clear reason for a customer's **RTP (Refuse to Pay)** outcome and compares transcript-based findings with the reason recorded by the agent. **PTP** means Promise to Pay. The transcript analysis focuses on what the customer and agent actually said: hardship, other reasons for non-payment, the agent's response, the call's outcome, and any explicit payment promise.

## Folders and files

- `cases/` — source case data, including `transcribe cases .csv` with call metadata and full conversations.
- `results/transcripts_for_ollama.csv` — transcript input prepared for the local Ollama analysis. It contains `task_id`, `user_id`, `call_info_id`, and `full_conversation`.
- `analysis/` — Python scripts for transcript analysis and comparison.
- `results/` — existing analysis outputs, review samples, summaries, and earlier Ollama experiments. Files in `results/experiments/` are exploratory outputs, not validated findings.

## Ollama transcript analysis

The Ollama scripts send each transcript to a model running locally through Ollama's `/api/generate` endpoint. They ask the model to return structured JSON, then save one result row per call to a CSV. The calls are Brazilian Portuguese collection conversations. Ollama must be running locally and the requested model must already be installed.

### `analysis/analyze_transcripts_ollama.py`

This is the single-pass version. For each call, it asks the model to classify customer sentiment and tone, stated hardship, other non-payment reasons, agent response, conversation result, and promise status. It also extracts a promised amount, a spoken payment date, product names, a short evidence quote, and confidence.

The script constrains the model's response to a JSON schema and checks the returned categories. It then applies grounding checks: for example, it verifies that the evidence quote appears in a transcript line, that a spoken date came from a customer turn, and that mentioned products appear in the conversation. It records issues in `review_issues` and lowers confidence when a check fails. These checks help flag unsupported output; they do not replace human review.

It requires a unique `task_id` for every input row and these input columns: `task_id`, `user_id`, `call_info_id`, and `full_conversation`. It appends results to the output CSV and skips `task_id`s already present there, so an interrupted run can be resumed. Failed calls are written with an `error` value.

Example (replace `your-installed-model` with a name shown by `ollama list`):

```bash
python3 analysis/analyze_transcripts_ollama.py \
  --input results/transcripts_for_ollama.csv \
  --output results/experiments/ollama_run.csv \
  --model your-installed-model \
  --limit 10
```

Remove `--limit 10` to process all remaining calls. Useful options include `--url` for a different Ollama server, `--timeout` for the per-call timeout, `--context` for the model context size, `--retries` for retrying failed calls, and `--think` to enable model reasoning (which may take longer). The script's default input is `analysis/transcripts_for_ollama.csv`; the prepared input currently in this project is under `results/`, so pass `--input` as shown above.

### `analysis/analyze_transcripts_ollama_v4.py`

This is an evidence-first variant. It uses two model requests per call: first it asks for short, exact quotes from the transcript, then it asks the model to classify the call using those quotes and the customer turns. Python checks that each quote appears in the appropriate speaker's transcript line. It derives amounts and dates from the quoted customer promise and applies additional guardrails, leaving unsupported details blank. This makes the output easier to audit, but it takes two model requests per call and is still a model-assisted analysis that needs review.

It accepts the same four input columns and has similar resume behavior, using `task_id` to skip rows already in the output. Example:

```bash
python3 analysis/analyze_transcripts_ollama_v4.py \
  --input results/transcripts_for_ollama.csv \
  --output results/experiments/ollama_evidence_first_run.csv \
  --model your-installed-model \
  --limit 10
```

The `--think`, `--timeout`, `--context`, and `--url` options are also available. Its default input is likewise `analysis/transcripts_for_ollama.csv`, so use the explicit input path for the prepared file in `results/`.

### `analysis/compare_agent_vs_transcript.py`

This script is intended to join agent-recorded call reasons with transcript-analysis results by `task_id`. It creates a case-level comparison, summary metrics, and a repeatable sample for human review. A candidate is flagged when an RTP call has a blank or generic agent reason but the transcript analysis identifies a potentially actionable reason; a stricter review flag also requires matching transcript evidence and medium or high model confidence.


## Reading the outputs

Treat model classifications and candidate flags as leads for review, not as confirmed facts. Check the transcript and the relevant evidence quote before using a result. The CSVs in `results/experiments/` are earlier trials; the repository does not designate them as a validated final model run.
