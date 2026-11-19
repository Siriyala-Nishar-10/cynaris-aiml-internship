# W7D4 — Ollama Setup: Local LLM Inference and Model Comparison

## What This Does

Installs Ollama, pulls two small local models (`llama3.2:3b` and `qwen2.5:3b`),
calls the Ollama REST API from Python with a custom system prompt, and compares
the two models on the same questions.

## Run

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
ollama run llama3.2:3b "Say hello"         # first local inference
python ollama_prompts.py                    # 5 prompts, llama3.2:3b
python ollama_prompts.py qwen2.5:3b         # 5 prompts, qwen2.5:3b
python compare_models.py                    # same 3 questions on both models
```

## Files

- `ollama_client.py` — small wrapper over `POST /api/chat` (system + user messages, timing and token stats)
- `ollama_prompts.py` — task 2: 5 prompts with a custom system prompt, saved to `results/prompts_<model>.csv`
- `compare_models.py` — task 3: llama3.2:3b vs qwen2.5:3b on 3 questions, saved to `results/model_comparison.csv`

## Design Decisions

| Decision               | Choice                                                                   | Why                                                             |
| ---------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------- |
| API                    | Ollama REST (`/api/chat`, `stream: false`)                               | shows the raw API call; no extra SDK needed                     |
| System prompt (task 2) | concise RAG-intern assistant, max 4 sentences, admit uncertainty         | tests whether the model follows constraints and avoids guessing |
| Prompt set             | explanation, comparison, summary, JSON extraction, unanswerable question | covers format following and hallucination behaviour             |
| Temperature            | 0.2 for the prompt tests, 0.0 for the model comparison                   | the comparison must be repeatable                               |
| Warm-up call           | one tiny call per model before the comparison                            | model load time would otherwise inflate the first answer        |
| Metrics                | wall time, output tokens, tokens/s                                       | compare speed as well as quality                                |

## Results

### Task 2 — 5 prompts

Both models answered all five prompts. Both admitted they could not know the
closing price (P5). Response times were 4 to 25 seconds, running on CPU.

| Prompt                  | llama3.2:3b                                                           | qwen2.5:3b                                                                   |
| ----------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| P1 explain RAG          | correct, 2 sentences                                                  | correct, 1 sentence                                                          |
| P2 BM25 vs dense        | correct, but claims dense is "generally more efficient and effective" | correct, more precise definitions                                            |
| P3 one-sentence summary | correct                                                               | correct                                                                      |
| P4 JSON extraction      | `{"number": 85, "count": 50}` (vague keys)                            | `{"correct_answers_percentage": 85, "total_queries": 50}` (descriptive keys) |
| P5 unanswerable         | admits it does not know, cites knowledge cutoff                       | admits it does not know                                                      |

### Task 3 — same 3 questions, temperature 0

| Question                                            | llama3.2:3b                                               | qwen2.5:3b                                                              |
| --------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------- |
| Factual: what does RAG stand for?                   | **Wrong** — "Reinforcement and Goal"                      | Correct — Retrieval-Augmented Generation, with the problem it addresses |
| Reasoning: 17/20 accuracy, next 30 questions        | **Wrong** — answers 527 (multiplies 17 by 30 and adds 17) | Correct — 0.85 x 30 = 25.5, rounded to 26                               |
| Instruction: exactly 3 numbered items, 12 words max | 3 items within the limit, but adds an intro line          | 3 items within the limit, no extra text                                 |

My ratings (1–5), please verify before submitting:

| Model       | Question    | Accuracy | Instruction following | Clarity |
| ----------- | ----------- | -------- | --------------------- | ------- |
| llama3.2:3b | factual     | 1        | 4                     | 4       |
| llama3.2:3b | reasoning   | 1        | 2                     | 2       |
| llama3.2:3b | instruction | 4        | 3                     | 4       |
| qwen2.5:3b  | factual     | 5        | 5                     | 5       |
| qwen2.5:3b  | reasoning   | 4        | 5                     | 5       |
| qwen2.5:3b  | instruction | 4        | 5                     | 5       |

Average ratings across the 3 questions: llama3.2:3b accuracy 2.0, instruction following 3.0, clarity 3.3; qwen2.5:3b accuracy 4.3, instruction following 5.0, clarity 5.0.

### Speed

|                  | llama3.2:3b        | qwen2.5:3b          |
| ---------------- | ------------------ | ------------------- |
| Generation speed | 7.9 – 9.4 tokens/s | 9.3 – 10.3 tokens/s |

The first call to each model is slow because Ollama loads the model into memory.
In the P1 run, qwen2.5:3b took 24.7 s for only 38 tokens, about 20 s of that
was model loading. That is why the comparison script makes a warm-up call first.

## Findings

- On this small test qwen2.5:3b was better: it answered the factual and
  reasoning questions correctly, while llama3.2:3b hallucinated the meaning of
  RAG and produced a wrong calculation.
- qwen2.5:3b followed the output format more strictly (no extra intro line,
  descriptive JSON keys) and generated slightly faster.
- Both models handled the unanswerable question well by admitting uncertainty,
  so the system prompt instruction to avoid guessing worked for both.
- Speed on CPU is slow (about 8 to 10 tokens/s), so local models suit
  low-volume or privacy-sensitive work more than interactive use.

**Limits:** 3 comparison questions, one run each. This is enough to show a
difference on these questions, not to rank the models in general.

## Notes

- Both models are quantised builds (about 2 GB each), which trades some quality
  for lower memory use and faster inference.
- Environment: Windows, Ollama, per-day `.venv`.

## If I Had One More Day

- Run each question several times at different temperatures and report variance
- Add a larger question set and score answers against a reference
- Compare a 7B model to see whether the quality gap closes
- Test a RAG prompt (context plus question) instead of bare questions

## Author

Siriyala Nishar
