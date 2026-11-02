# W7D3 — LlamaIndex Document Indexing & Querying

## What This Does

Extends the document-question-answering workflow over 5 PDF research papers. The same 10 questions are run through two retrieval modes, each using a LlamaIndex query engine with Ollama embeddings:

- **Memory** — a LlamaIndex `VectorStoreIndex` using the in-memory vector store
- **Chroma** — a LlamaIndex `VectorStoreIndex` backed by ChromaDB

The generated answers are checked against the source papers. Each results CSV records the answer, retrieval time, total query time, retrieved source files, and the `correct` label (1 = correct, 0 = incorrect or incomplete).

Pipeline: PDF-to-text conversion -> document loading and indexing -> embedding and vector retrieval -> LLM answer generation -> answer verification

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

ollama pull nomic-embed-text
ollama pull llama3.2

python pdf_to_txt.py
python rag_compare.py memory      # results/memory_results.csv
python rag_compare.py chroma      # results/chroma_results.csv

# Check the `correct` column (1/0) in each CSV, then:
python summarise.py
```

Make sure Ollama is running before executing the query scripts. If using a different Ollama chat model, set `LLM_MODEL` to the installed model name.

## Data

- `pdfs/` — 5 source research papers (not committed; see Notes).
- `data/` — extracted `.txt` files used for indexing. If text files already exist, PDF conversion can be skipped.
- `questions.txt` — the 10 evaluation questions used for both modes.
- `results/` — generated CSV files for the memory and Chroma runs.

The papers cover GrowWise SME financial advisory, mushroom species classification using CNNs, RAG-based intelligent banking assistants, agentic AI in finance, and AI-powered business intelligence for financial services.

## Design Decisions

| Component | Choice | Why |
| --------- | ------ | --- |
| PDF conversion | `pdf_to_txt.py` | Creates text files for indexing |
| Embeddings | Ollama `nomic-embed-text` | Converts document chunks and queries into embeddings |
| Language model | Ollama `llama3.2` by default | Generates answers from retrieved context |
| Index | LlamaIndex `VectorStoreIndex` | Provides document indexing and querying |
| Memory mode | In-memory vector store | Baseline without a persistent Chroma collection |
| Chroma mode | ChromaDB vector store | Compares retrieval using a persistent vector-store backend |
| Evaluation | Same 10 questions in both modes | Keeps the question set consistent |
| Timing | `retrieval_s` and `total_s` | Separates retrieval timing from total answer-generation timing |

Only the vector-store backend changes between the two modes in the intended comparison. Model generation, question set, and source documents should remain consistent for a fair comparison.

## Results

Measured locally on the recorded run:

| Metric | Memory | Chroma |
| ------ | ------: | ------: |
| Index build time (s) | 125.02 | 118.73 |
| Mean retrieval time (s) | 2.406 | 0.708 |
| Median retrieval time (s) | 2.160 | 0.103 |
| Mean total query time (s) | 44.89 | 4.64 |
| Median total query time (s) | 45.16 | 3.90 |
| Correct answers | 9/10 | 9/10 |

Output evidence: capture the terminal output of `python summarise.py` in the project’s `docs/` folder if required by the submission.

## Findings

- Both modes achieved **9/10 correct answers** in the recorded evaluation.
- Chroma recorded a slightly lower index build time: **118.73 s**, compared with **125.02 s** for memory.
- Chroma also recorded lower mean and median retrieval times in this run.
- The recorded mean total query time was **44.89 s** for memory and **4.64 s** for Chroma. This difference is large enough that it should be checked with repeated runs before drawing a firm conclusion about typical performance.
- These are measurements from one local run. Hardware, model configuration, caching, and execution conditions can affect the timings; the results do not establish that one backend will always be faster.

**Limits of this evidence:** 10 fact-lookup questions across 5 documents, with answer correctness labelled against the source papers. The result is a small evaluation set and should be treated as indicative rather than conclusive.

## Issues Faced

- **Answer verification.** Generated answers needed to be checked against the original source papers, not judged only by whether the response sounded plausible.
- **Timing interpretation.** `retrieval_s` measures query embedding plus vector search, while `total_s` also includes LLM generation. These measurements represent different parts of the pipeline.
- **Run-to-run variation.** The substantial difference in total query times means repeated runs under consistent conditions would help confirm whether the timing pattern is stable.

## Notes

- The research PDFs are copyrighted source papers and should not be committed. Keep `pdfs/` and extracted text files excluded from version control.
- Keep `.venv/` and generated `chroma_db/` data out of Git.
- Environment shown in the provided terminal: Windows PowerShell with a Python virtual environment. Exact package versions were not included in the shared output.
- Do not report the timings as a general benchmark; they are specific to this run.

## What's Implemented

- `pdf_to_txt.py` — converts source PDFs into text files.
- `rag_compare.py` — runs the memory and Chroma query modes.
- `questions.txt` — the 10 evaluation questions.
- `summarise.py` — summarizes build time, retrieval latency, total query time, and correct-answer counts.
- `results/` — intended location for `memory_results.csv` and `chroma_results.csv`.

## If I Had One More Day

- Repeat both modes several times under the same conditions and compare averages and medians.
- Inspect the per-question CSVs to understand the incorrect or incomplete answer in each mode.
- Add more evaluation questions to make the comparison less dependent on a small sample.
- Record exact Python and package versions for reproducibility.

## Author

Siriyala Nishar
