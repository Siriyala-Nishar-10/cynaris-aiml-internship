# W5D6: LLM Cost Optimisation, Token Economics & Caching

Cost analysis and optimisation for the local Q&A bot from W5D5 (Ollama + ChromaDB). The local model costs nothing per call, so each request is also priced as if it had gone to a paid API (GPT-4o and GPT-4o mini). That shows what the workload would cost at scale, and which of the lesson's techniques (prompt compression, semantic caching, model routing) would cut it.

## What's in this folder

| File                                                          | Purpose                                                                                         |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `token_audit.py`                                              | Task 1: token counts of the bot's prompt at 5 sizes (tiktoken), priced for 4 API models in INR  |
| `compress_prompt.py`                                          | Task 2: LLMLingua compression at rate 0.4 on a ~2,000-token RAG prompt, quality vs uncompressed |
| `semantic_cache.ipynb`                                        | Task 3: semantic cache with 20 Q&A pairs, cosine similarity threshold 0.92, threshold sweep     |
| `cost_tracker.py`                                             | Task 4: logs tokens per request and reports the daily total in INR                              |
| `routing_strategy.md`                                         | Task 5 and deliverable 3: which queries go local, to a cheap API, or to a strong API            |
| `build_audit_xlsx.py`                                         | Builds `token_cost_audit.xlsx` (deliverable 1) from the measured files below                    |
| `token_cost_audit.xlsx`                                       | Token-cost audit: baseline vs optimised spend, with live formulas                               |
| `token_audit.csv`, `compression_results.json`, `cost_log.csv` | Measured data used by the spreadsheet                                                           |
| `qa_bot.py`                                                   | Copied from W5D5; the scripts import it for retrieval and the system prompt                     |
| `evidence/`                                                   | Screenshots of outputs                                                                          |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Mac/Linux: source .venv/bin/activate
pip install chromadb pypdf ollama numpy tiktoken llmlingua openpyxl ipykernel

ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Put a text-based PDF in `docs/` (same as W5D5) and keep Ollama running.

## Run order

```bash
python token_audit.py                 # Task 1  -> token_audit.csv
python compress_prompt.py             # Task 2  -> compression_results.json
# open semantic_cache.ipynb in VS Code, select the .venv kernel, Run All   (Task 3)
python cost_tracker.py demo           # Task 4  -> cost_log.csv
python cost_tracker.py report
python build_audit_xlsx.py            # Deliverable 1 -> token_cost_audit.xlsx
```

## Results

### Task 1: token audit

The bot's prompt (system prompt + retrieved context + question) was measured with 1, 3, 5, 8 and 12 retrieved chunks, using tiktoken (GPT-4o's tokenizer).

Using existing index: 45 chunks (use --rebuild to refresh)

Question: What does the input data of the RAG system comprise?
Assumed output: 150 tokens | USD->INR 88.0

size chars tokens GPT-4o Claude Sonnet 4 Gemini 2.0 Flash GPT-4o mini (cost per request, INR)
1 chunks 1233 228 0.2983 0.2582 0.0073 0.0109
3 chunks 2871 537 0.4343 0.3398 0.0100 0.0150
5 chunks 4509 911 0.5988 0.4385 0.0133 0.0199
8 chunks 6966 1330 0.7832 0.5491 0.0170 0.0255
12 chunks 10242 1963 1.0617 0.7162 0.0226 0.0338

Saved token_audit.csv
VERIFY: tokens should grow roughly linearly with the number of chunks (about 4 characters per token)

label,k_chunks,chars,tokens_tiktoken,tokens_ollama
1 chunks,1,1233,228,
3 chunks,3,2871,537,
5 chunks,5,4509,911,
8 chunks,8,6966,1330,
12 chunks,12,10242,1963,

Cost = input tokens x input price + output tokens x output price, using the lesson's prices and ₹88 per USD.

### Task 2: LLMLingua compression (rate = 0.4)

- **Compression:** a ~2,080-token context was reduced to ~810 tokens on average (kept fraction **0.389**, about 61% removed), measured over 3 questions.
- **Speed:** with the compressed context the local model answered in about 38 s on average instead of about 113 s (roughly 3x faster), because prompt processing dominates on CPU.
- **Quality dropped sharply:** average keyword recall fell from **0.89** (full context) to **0.39** (compressed).

| Question                     | Tokens before → after | Recall full → compressed | Seconds full → compressed |
| ---------------------------- | --------------------- | ------------------------ | ------------------------- |
| Main topic                   | 2,087 → 809           | 0.67 → 0.67              | 126.7 → 39.9              |
| Two phases of the research   | 2,035 → 801           | 1.00 → 0.00              | 99.0 → 32.3               |
| Input data of the RAG system | 2,125 → 820           | 1.00 → 0.50              | 112.6 → 42.4              |

Manual comparison against the PDF:

- **Main topic:** the recall score is equal, but the compressed answer is wrong. It expands RAG as "Reasoning About Genres", while the full-context answer correctly describes the HR banking assistant. The keyword score missed this, which is why the manual check matters.
- **Two phases:** the full context gave both phases correctly (Excel question-answer pairs, then unstructured PDFs). The compressed context lost those facts and the model answered "I don't know based on the documents."
- **Input data:** the full answer listed employee queries and structured internal data. The compressed answer described a different detail (a dataset of 100 Excel question-answer pairs) and also contained a stray "[3315]".
- **Citations:** the compressed answers cited `file.pdf` instead of `sample.pdf`, so the `[sample.pdf p.N]` source tags were damaged by the compression.
- **Conclusion:** at rate 0.4 the token saving (about 2.6x fewer tokens) came with far more quality loss than the lesson's "under 5%" figure for RAG tasks. Likely causes are the aggressive rate, a small 3B model reading a compressed context, and compression applied to the source tags as well as the text. A higher rate (for example 0.6 to 0.7) and keeping the source tags out of the compressed text would be the next things to test.
- LLMLingua-2 (a small BERT-based compressor) was used so that it runs on CPU, with `device_map="cpu"` because the PyTorch build has no CUDA.

### Task 3: semantic cache (cosine similarity threshold 0.92)

20 Q&A pairs were embedded with `nomic-embed-text`. Ten paraphrases and five unrelated questions were tested.

| Threshold | Correct hits (of 10) | Wrong hits | Unrelated hits (of 5) |
| --------- | -------------------- | ---------- | --------------------- |
| 0.80      | 8                    | 0          | 0                     |
| 0.84      | 6                    | 0          | 0                     |
| 0.88      | 5                    | 0          | 0                     |
| 0.90      | 4                    | 0          | 0                     |
| **0.92**  | **4**                | **0**      | **0**                 |
| 0.94      | 4                    | 0          | 0                     |
| 0.96      | 2                    | 0          | 0                     |

- At the lesson's 0.92 the hit rate was **40%**, with no wrong hits. Unrelated questions scored only 0.39 to 0.43.
- Paraphrases that use different wording miss: "What does retrieval-augmented generation mean?" scored 0.580 against the cached "What is RAG?".
- Live demo: a paraphrase of a cached question was served from the cache (similarity 0.966), and a repeated new question was served from the cache on its second ask (1.000), so 3 questions needed only 1 LLM call.
- Recommended starting threshold: **0.88**, then lower it carefully after testing on-topic questions with different meanings, which this test set doesn't cover.
- Estimated saving at a 40% hit rate: about ₹6.34 per 1,000 queries on GPT-4o mini and ₹184.80 per 1,000 on GPT-4o (600 input and 150 output tokens per call).

### Task 4: cost tracker

Five questions went through the bot and were logged with Ollama's real token counts:

| Question                      | Prompt tokens | Output tokens | Seconds |
| ----------------------------- | ------------- | ------------- | ------- |
| Main topic                    | 545           | 34            | 19.47   |
| Problem addressed             | 547           | 129           | 20.91   |
| Two phases                    | 674           | 75            | 23.33   |
| Input data of the RAG system  | 568           | 72            | 19.80   |
| Out of scope (2010 World Cup) | 549           | 42            | 16.15   |

Daily report (2026-10-03): 5 requests, 2,883 prompt tokens, 352 output tokens.

|                             | Cost for 5 requests |
| --------------------------- | ------------------- |
| Actual (local inference)    | ₹0.00               |
| Same traffic on GPT-4o mini | ₹0.0566             |
| Same traffic on GPT-4o      | ₹1.7332             |

GPT-4o would cost about 31 times more than GPT-4o mini for the same requests. Extrapolated to the lesson's 10,000 queries per day, that is about ₹113 per day on GPT-4o mini against about ₹3,466 per day on GPT-4o.

### Task 5 and deliverable 3: routing strategy

`routing_strategy.md` proposes four tiers: semantic cache, local model, cheap API and strong API. Routing uses cheap signals computed before any LLM call (cache similarity, retrieval similarity, question complexity, context size, latency budget). Sensitive documents never leave the machine. The strategy is based on the W5D5 findings, such as the 5 to 45 second CPU latency and the small model's citation errors.

### Deliverable 1: token-cost audit spreadsheet

`token_cost_audit.xlsx` has two sheets:

- **Assumptions:** prices, exchange rate, traffic, compression ratio (measured, 0.389), output tokens per answer (measured, 70), cache hit rate and cheap-model share. Yellow cells are assumptions to replace.
- **Audit:** per prompt size, the baseline cost (every query to GPT-4o with the full prompt) against the optimised cost (compression + semantic cache + cheap-model routing), with the saving in % and in INR per month. Average saving across the 5 prompt sizes: [read from the Audit sheet, Average row, column K].

## Limitations

- Token counts use GPT-4o's tokenizer, so they are estimates for other models. Ollama's own counts for the local model differ and can be lower when it reuses a cached prompt prefix.
- The cache hit rate comes from a small, hand-written test set, not real traffic. Wrong hits are 0 here, but the test set has no on-topic questions with a different meaning, which is where wrong hits are most likely.
- The routing strategy is a proposal. It isn't implemented in code or tested on labelled queries.
- The spreadsheet's optimised cost depends on assumptions (cache hit rate, 70% cheap-model share from the lesson, traffic and exchange rate). Prices come from the lesson table, so check them against current provider pricing before quoting them.
- Inference on CPU is slow (16 to 23 seconds per request), which matters for routing latency-sensitive queries.

## Self-review checklist

- [x] Token audit run at 5 prompt sizes and priced
- [x] LLMLingua at rate 0.4 run; answers compared against the PDF
- [x] Semantic cache notebook run with clean outputs; Observations filled in
- [x] Cost tracker logs tokens per request and reports the daily total in INR
- [x] Routing strategy written
- [x] Spreadsheet opened in Excel; assumptions replaced with measured values
- [x] `.venv/`, `chroma_db/`, `docs/` in `.gitignore`

## Author

Siriyala Nishar
