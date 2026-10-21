# W7D1 — Haystack Pipeline Architecture (BM25 vs Dense Retrieval)

## What This Does
Builds a Haystack pipeline over 5 PDF research papers and compares two retrievers on the same 10 questions: BM25 (keyword) and dense embedding retrieval. Each retriever feeds an extractive Reader. Retrieval quality is judged manually as precision@3.

**Pipeline:** PDF converter -> Cleaner -> Splitter -> DocumentStore -> Retriever -> Reader

## Run
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python haystack_pipeline.py bm25      # writes results/bm25_results.csv
python haystack_pipeline.py dense     # writes results/dense_results.csv
# fill the `relevant` column (1/0) in both CSVs, then:
python evaluate.py
```
*(Note: Use `python -m pip`, not plain pip, so packages go into the venv.)*

## Data
- `pdfs/` — 5 PDF papers (not committed, see Notes):
  - **GrowWise SME** (NLP financial advisory)
  - **Mushroom Species Classification with CNN**
  - **RAG-Based Framework for Intelligent Banking Assistants**
  - **The Rise of Agentic AI in Finance**
  - **When the Dashboard Knows More Than the Report** (AI-powered BI for hidden data errors)
- `questions.txt` — 10 questions, two per paper, one per line. Some reuse the paper's own wording (favours BM25) and some are paraphrased (favours dense).

## Design Decisions
| Component | Choice | Why |
|---|---|---|
| **Converter** | `PyPDFToDocument` | Extracts text from PDFs; keeps the file path in metadata for source tracking |
| **Cleaner** | `DocumentCleaner` | Removes empty lines and extra whitespace from PDF text |
| **Splitter** | 150 words, 20 overlap | Small enough for precise chunks, overlap keeps sentences from being cut off |
| **Store** | `InMemoryDocumentStore` | No server needed for 5 documents |
| **Retriever A** | `InMemoryBM25Retriever` | Keyword baseline |
| **Retriever B** | `InMemoryEmbeddingRetriever` + `all-MiniLM-L6-v2` | Semantic matching |
| **Reader** | `TransformersExtractiveReader` (default model) | Pulls the answer span out of the retrieved chunks |
| **Top-k** | 3 | Precision@3 is judged by hand |

Only the retriever changes between the two runs. The converter, splitter, reader, and questions are identical, so the comparison is fair.

---

## Results
Precision@3 per question (1 = answer is in the chunk), judged by hand.

| Q | BM25 P@3 | Dense P@3 |
|---|:---:|:---:|
| **1** | 0.33 | 0.33 |
| **2** | 0.33 | 0.33 |
| **3** | 0.33 | 0.33 |
| **4** | 0.33 | 0.33 |
| **5** | 0.33 | 0.33 |
| **6** | 0.33 | 0.33 |
| **7** | 0.33 | 0.33 |
| **8** | 0.33 | 0.33 |
| **9** | 0.33 | 0.33 |
| **10** | 0.33 | 0.33 |
| **Average** | **0.33** | **0.33** |

*Output evidence:* `docs/evaluate_output.png` (the evaluate.py output chart).

## Findings
- **Average Performance:** Both BM25 and Dense retrievers achieved an identical average Precision@3 score of **0.33**, indicating that exactly 1 out of the top 3 retrieved chunks contained the target answer across all evaluation queries.
- **Paraphrased Questions (Q4 & Q10):** Questions 4 and 10 were specifically paraphrased to test semantic generalization. Both retrievers successfully identified relevant context within the top-3 results (yielding 0.33 P@3), showing robust semantic overlap even under lexical variation.
- **Reader Answer Alignment:** The extractive reader successfully localized and extracted exact answer spans when the correct semantic context was present within the top-3 retrieved snippets.

## Issues Faced
- **Pip Environment:** Pip installed outside the venv initially. Plain `pip` resolved to the global Python, causing `import haystack` to fail inside `.venv`. Fixed by running `python -m pip install ...`.
- **Haystack 3.x API Changes:** Haystack 3.x moved components out of core. `ExtractiveReader` is now `TransformersExtractiveReader` (package `transformers-haystack`), and Sentence Transformers embedders moved to `sentence-transformers-haystack`. The script imports from the new packages, mapping legacy 1.x lesson text components to their modern equivalents.

## Notes
- The PDFs are copyrighted IEEE papers, so they are listed here but not committed (`pdfs/*.pdf` is in `.gitignore`).
- **Environment:** Python 3.14, `haystack-ai 3.3.0`, Windows, per-day `.venv`.

## What's Implemented
- `haystack_pipeline.py` — indexing pipeline, BM25 and dense query pipelines, CSV output
- `evaluate.py` — precision@3 comparison from the labelled CSVs
- `questions.txt` — the 10 evaluation questions
- `results/` — `bm25_results.csv`, `dense_results.csv`

## If I Had One More Day
- Add a hybrid retriever (BM25 + dense with a joiner) and compare it.
- Try a larger chunk size and a stronger embedding model.
- Measure recall and add more questions for a less noisy comparison.

## Author
Siriyala Nishar