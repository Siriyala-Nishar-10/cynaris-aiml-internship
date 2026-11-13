# W7D5 — Week 7 Project: Multi-document RAG System

## What This Does

A multi-document RAG system over 5 research-paper text files, built with
LlamaIndex and local Ollama models, with experiment results logged to MLflow.
Two designs are compared on the same 10 questions:

- **flat** — one `VectorStoreIndex` over all documents (the Day 3 design)
- **router** — one index per paper; a `RouterQueryEngine` with an LLM selector
  reads a one-line description of each paper and sends the question to the right one

## Run

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q                  # unit tests for the helper functions
ollama pull nomic-embed-text
ollama pull llama3.2
python pdf_to_txt.py                 # PDFs in ./pdfs -> .txt in ./data
python multi_doc_rag.py flat
python multi_doc_rag.py router
# verify each answer against the paper, fill the `correct` column (1/0), then:
python summarise.py                  # comparison table + MLflow logging
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Files

- `multi_doc_rag.py` — flat and router RAG, per-question CSV and meta output
- `utils.py` — expected paper per question, paper descriptions, answer key, helpers
- `tests/test_utils.py` — unit tests for the helpers
- `summarise.py` — comparison table and MLflow logging (experiment `w7d5-multidoc-rag`)
- `results/` — `flat_results.csv`, `router_results.csv`

## Design Decisions

| Decision | Choice | Why |
| -------- | ------ | --- |
| Router | `RouterQueryEngine` + `LLMSingleSelector` | works with a small local model; no function calling needed |
| Tool descriptions | hand-written, one line per paper | the selector chooses based on these, so they name each paper's topic |
| Per-paper index | one `VectorStoreIndex` per file | a question searches only its own paper |
| Chunking / top-k | 512 tokens, 50 overlap, top-3 | same as Day 3, so results are comparable |
| Error handling | a failed routing call is caught and counted as an error | small models sometimes return malformed selections |
| Evaluation | manual `correct` label plus two automatic checks | manual is the reported number; the checks catch mistakes |

Automatic checks: `top_doc_ok` (was the first retrieved chunk from the right
paper?) and `keyword_hit` (does the answer contain the key fact?).

## Results

| Metric | Flat | Router |
| ------ | ---- | ------ |
| Correct answers (manual) | 8/10 | 9/10 |
| Keyword check | 8/10 | 9/10 |
| Right paper retrieved first | 10/10 | 10/10 |
| Routing errors | – | 0 |
| Latency mean / median | 45.8 s / 45.9 s | 12.7 s / 10.7 s |
| Index build time | [from summarise.py] | [from summarise.py] |

Evidence: `docs/summarise_output.png` and `docs/mlflow_runs.png`.

## Findings

- The router scored 9/10 against 8/10 for flat. The only difference was Q2:
  flat described the recommendation logic without naming the techniques, while
  the router's answer mentioned cosine similarity. One question is a small
  difference, so the two designs are close.
- Both designs retrieved the right paper first on every question, so routing did
  not decide the result here. With five clearly different papers, a flat index
  is already good at finding the right one; routing would matter more with many
  similar documents.
- Both failed Q7 (Wolters Kluwer survey): the right paper was retrieved but the
  top-3 chunks missed the sentence with the 6% figure, and the model answered
  with an unrelated "sixfold increase". This is a chunk-level retrieval miss,
  the same weakness seen with the Haystack runs on Day 2.
- The router had no routing errors, so the local llama3.2 model handled the
  selector prompt reliably on this question set.
- The latency gap should not be read as "routing is 4x faster". It needs a
  controlled rerun: on Day 3 two runs with the same retrieval logic differed by
  about 10x because of Ollama's model loading, and the router runs were likely
  warmer. Compare medians with both modes run back-to-back.

**Limits:** 10 questions, 5 documents, one run per design, labels judged by one
person. Partial answers that state the key fact (Q1, Q9, Q4) were counted correct.

## Notes

- The source PDFs are copyrighted IEEE papers; they and the converted text files
  are not committed (`.gitignore`), nor are `mlflow.db` or `mlruns/`.
- Environment: Python 3.14, Windows, Ollama (`llama3.2`, `nomic-embed-text`).

## If I Had One More Day

- Raise top-k or use hybrid retrieval to fix the Q7-style chunk misses
- Re-run flat and router back-to-back with warm models to get a fair latency comparison
- Add a Ragas faithfulness and relevance score with a stronger judge model
- Test the router with more, and more similar, documents

## Author

Siriyala Nishar
