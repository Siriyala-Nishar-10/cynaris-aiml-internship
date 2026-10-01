# W5D1: Running LLMs Locally with Ollama — Setup & First Run

## Objective

Install Ollama, pull `llama3.2:3b`, run local inference, call the Ollama
REST API from Python with a custom system prompt (5 test prompts), and
compare `llama3.2:3b` against `qwen2.5:3b` on the same 3 questions.

## What's in this folder

- `ollama_client.py` — thin client for Ollama's REST API (`/api/tags`, `/api/chat`) with typed errors and timing stats
- `run_prompts.py` — **Task 2**: custom system prompt, 5 test prompts
- `compare_models.py` — **Task 3**: same 3 questions on both models, with automatic checks and a generated report
- `test_ollama_client.py` — 13 tests against a mock Ollama server (no real Ollama needed)
- `requirements.txt`
- `run_prompts_results.json` — output of Task 2 _(generated when you run it)_
- `comparison_results.json`, `comparison_report.md` — output of Task 3 _(generated when you run it)_

## Setup (Windows)

1. **Install Ollama:** download from https://ollama.com/download and run the installer. It starts a background server automatically (system tray icon) on `http://localhost:11434`.
2. **Verify:** `ollama --version`
3. **Pull the models** (each is a multi-GB download):
   ```
   ollama pull llama3.2:3b
   ollama pull qwen2.5:3b
   ```
4. **First inference (Task 1):** `ollama run llama3.2:3b`, type a question, then `/bye` to exit. Screenshot this as evidence.
5. **Inspect the model:** `ollama show llama3.2:3b` prints parameter count, context length and quantisation level (useful for the viva).
6. **Python deps:** `pip install -r requirements.txt`

## Run

```
python -m unittest test_ollama_client -v     # tests (mock server, no Ollama needed)
python run_prompts.py                        # Task 2
python compare_models.py                     # Task 3 (writes comparison_report.md)
```

## Design decisions

- **Raw REST via `requests`** instead of the `ollama` pip package: shows exactly what goes over the wire (messages, options, timing fields). The package wraps the same endpoints.
- **Low temperature + fixed seed** (`temperature=0.2, seed=42`) so runs are mostly repeatable — important when comparing two models or re-running for evidence.
- **Warm-up call before timing** in the comparison: the first request loads the model into memory, which would otherwise make whichever model runs first look slow.
- **Tokens/sec from Ollama's own `eval_count / eval_duration`**, which excludes model-load time, alongside wall-clock latency.
- **The 3 comparison questions each test something different:** factual recall (small models often hallucinate names/dates), arithmetic reasoning (single verifiable answer), and instruction following ("exactly 3 bullets, each under 20 words").
- **Automatic checks are a rough signal, not a verdict.** They're substring/line-count tests, so a correct answer worded unusually can fail, and a passing check doesn't mean the answer is good. The written comparison below relies on reading the actual responses.
- **Typed errors with actionable messages:** if Ollama isn't running or a model isn't pulled, the script says exactly what to run.

## Testing

`test_ollama_client.py` starts a fake HTTP server that mimics Ollama's response shapes (including its 404 for a missing model) and verifies request building (system prompt ordering, options forwarding), response parsing, the ns→s and tokens/sec maths, error handling (missing model, HTTP 500, connection refused), the answer-checking helpers, and report generation. **These tests verify the client code, not any model's output** — model behaviour can only be observed by running against real Ollama.

## Results

**Model info** (`ollama show <model>`):

| Model       | Parameters | Quantisation | Context length |
| ----------- | ---------- | ------------ | -------------- |
| llama3.2:3b | 3.2B       | Q4_K_M       | 131072         |
| qwen2.5:3b  | 3.1B       | Q4_K_M       | 32768          |

**Comparison summary** (the table from `comparison_report.md`):

=== llama3.2:3b ===
[Q1 factual] check FAIL | 9.9 tok/s | 5.8s
[Q2 reasoning] check PASS | 9.3 tok/s | 11.4s
[Q3 instruction-following] check PASS | 9.1 tok/s | 9.8s

=== qwen2.5:3b ===
[Q1 factual] check FAIL | 10.3 tok/s | 7.4s
[Q2 reasoning] check PASS | 9.6 tok/s | 20.8s
[Q3 instruction-following] check PASS | 9.5 tok/s | 7.8s

**Differences I observed** (from reading the full responses):

- Accuracy / hallucination: Both models failed the factual question and
  hallucinated. Llama attributed The Discovery of India to Mahatma Gandhi
  (1927); Qwen attributed it to Indira Gandhi (1983) and also invented an
  alternate title ("My Life"). The correct answer is Jawaharlal Nehru, 1946.
  Neither model hedged or said it wasn't sure.
- Reasoning quality: Both got the correct answer (80 km/h) with correct
  working: convert 45 minutes to 0.75 hours, then distance / time. Llama was
  more concise (53 words); Qwen was more detailed (86 words) and used LaTeX
  formatting.
- Instruction following: Both passed the automatic check for exactly 3
  bullets under 20 words. Qwen followed the format directly. Llama added an
  intro sentence before the bullets, which the check ignores, so the check
  is more lenient than a strict reading. One of Qwen's bullets described a
  mitigation rather than defining overfitting.
- Style / verbosity: Llama was shorter on Q1 and Q2 (15 and 53 words vs 29 and
  86), but longer on Q3 (50 vs 33) because of its intro line. Qwen's Q1 answer
  was more elaborate and contained more fabricated detail.
- Speed on my hardware: Both generated at about 12-14 tokens/sec (Llama
  12.3-13.2, Qwen 12.0-13.6), a difference of under 5% on every question. That
  is within run-to-run noise: an earlier run recorded 9-10 tok/s for the same
  models. Qwen's longer wait on Q2 came from writing a longer answer, not from
  generating slower. CPU-only laptop, 16 GB RAM.

## Self-review checklist

- [done] Ollama installed; `llama3.2:3b` and `qwen2.5:3b` pulled
- [done] First `ollama run` inference done and screenshotted
- [done] `python -m unittest test_ollama_client -v` passes
- [done] `run_prompts.py` ran on all 5 prompts; output saved
- [done] `compare_models.py` ran; `comparison_report.md` observations section filled in by hand
- [done] Results table above filled with real numbers
- [done] Code commented; committed in 2+ commits; PR raised

## Author

Siriyala Nishar
