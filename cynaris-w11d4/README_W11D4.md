# W11D4 — RAG Evaluation with Ragas: Metrics & Benchmarks

## Objective

Build a LangChain + ChromaDB RAG pipeline, evaluate it with Ragas, tune chunking/retrieval, and compare metrics.

## Setup (Windows PowerShell)

Requires Python 3.11, Ollama running, and these local models:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
cd C:\Users\<your-user>\Desktop\cynaris-w1d1\cynaris-w11d4
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked, use `.\.venv\Scripts\python.exe` to run pip/python.

## Run

```powershell
python generate_qa.py
python -m pytest -q
python compare.py
```

Ragas evaluates faithfulness, answer_relevancy, context_precision, and context_recall. The evaluation uses local Ollama models; first run may take time.

## Experiment settings

| Setting         | Baseline | Tuned |
| --------------- | -------: | ----: |
| Chunk size      |      450 |   300 |
| Chunk overlap   |       60 |    60 |
| Retrieval top-k |        2 |     3 |

The workflow compares both settings. Report actual scores from `results/`; do not claim optimization improved a metric unless the measured score confirms it.

## Output files

`results/` contains generated Q&A, baseline/optimized answers, Ragas score CSVs, JSON summaries, and `metric_comparison.csv`. ChromaDB persists locally under `.chroma/` (ignored by Git).

## Metric guide

- Faithfulness: whether answer claims are supported by retrieved context.
- Answer relevancy: whether the answer addresses the question.
- Context precision: how much of the retrieved context is relevant.
- Context recall: whether retrieved context covers the information needed by the reference answer.
