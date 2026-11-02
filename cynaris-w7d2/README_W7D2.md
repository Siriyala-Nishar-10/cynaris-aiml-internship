# W7D2 — Haystack Retrieval: BM25 vs Dense vs Hybrid

## What This Does

Extends the Day 1 pipeline over 5 PDF research papers. The same 10 questions
are run through three retrieval modes, each feeding an extractive Reader:

- **BM25** — keyword retrieval
- **Dense** — embedding retrieval (all-MiniLM-L6-v2)
- **Hybrid** — BM25 and dense results merged with reciprocal rank fusion (RRF)

Retrieval quality is judged by hand as precision@3: for each question, how many
of the top 3 chunks actually contain the answer.

Pipeline: PDF converter -> Cleaner -> Splitter -> DocumentStore -> Retriever(s) -> Reader

## Run

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python haystack_pipeline.py bm25      # results/bm25_results.csv
python haystack_pipeline.py dense     # results/dense_results.csv
python haystack_pipeline.py hybrid    # results/hybrid_results.csv
# fill the `relevant` column (1/0) in each CSV, then:
python evaluate.py
```

## Data

- `pdfs/` — 5 PDF papers (not committed, see Notes): GrowWise SME (NLP financial
  advisory), Mushroom Species Classification with CNN, RAG-Based Framework for
  Intelligent Banking Assistants, The Rise of Agentic AI in Finance, and When
  the Dashboard Knows More Than the Report.
- `questions.txt` — 10 questions, two per paper. Some reuse the paper's wording
  (favours BM25) and some are paraphrased (favours dense).

## Design Decisions

| Component | Choice | Why |
| --------- | ------ | --- |
| Converter | `PyPDFToDocument` | extracts PDF text and keeps the file path for source tracking |
| Splitter | 150 words, 20 overlap | small, precise chunks; overlap avoids cutting sentences |
| Store | `InMemoryDocumentStore` | no server needed for 5 documents |
| BM25 | `InMemoryBM25Retriever` | keyword baseline |
| Dense | `InMemoryEmbeddingRetriever` + `all-MiniLM-L6-v2` | semantic matching |
| Hybrid | `DocumentJoiner` (reciprocal rank fusion) | each retriever returns 5 candidates, the joiner keeps the best 3 |
| Reader | `TransformersExtractiveReader` | extracts the answer span from the retrieved chunks |
| Top-k | 3 | precision@3 is judged by hand |

Only the retrieval part changes between modes. The converter, splitter, reader
and questions are identical, so the comparison is fair.

## Results

Precision@3 per question (hand-labelled against the full 150-word chunk text;
1 = chunk contains the answer).

| Q | BM25 | Dense | Hybrid |
| - | ---- | ----- | ------ |
| 1 | 0.67 | 0.00 | 0.67 |
| 2 | 0.33 | 0.33 | 0.33 |
| 3 | 0.00 | 0.67 | 0.33 |
| 4 | 0.33 | 0.33 | 0.33 |
| 5 | 0.33 | 0.33 | 0.33 |
| 6 | 1.00 | 0.33 | 0.67 |
| 7 | 0.33 | 0.33 | 0.33 |
| 8 | 0.67 | 0.67 | 0.67 |
| 9 | 1.00 | 0.33 | 0.67 |
| 10 | 0.67 | 0.67 | 0.67 |
| **Average** | **0.53** | **0.40** | **0.50** |

Output evidence: `docs/evaluate_output.png` (the `evaluate.py` output).

## Findings

- BM25 scored highest on average (0.53), hybrid was close (0.50) and dense was
  lowest (0.40). The BM25-hybrid gap is one relevant chunk in 10 questions, so
  they are effectively tied; dense is clearly behind on this question set.
- BM25 won on questions that hinge on exact terms or numbers, such as Q6 (the
  "85%" correct-answer rate) and Q9 ("203" vs "147" discrepancies), where the
  question words appear verbatim in the answer chunk.
- Dense won on Q3 (0.67 vs 0.00 for BM25), where the answer is spread across
  results text that shares meaning but not the question's wording. It failed
  completely on Q1 (0.00), retrieving general chunks instead of the one with
  the precision figures.
- Hybrid never scored 0 on any question, and it is never far from the best
  retriever. It trades a little peak precision for robustness, which is its
  main benefit when you do not know in advance which question style will come.
- The Reader is a weak point. On Q5 it answered "HierarchicalSplitter" from a
  chunk that lists the candidate splitters before saying which one was chosen.
- Several answers are split across neighbouring 150-word chunks (for example
  Q5, Q7 and Q9), which caps precision@3 for every retriever.

**Limits of this evidence:** 10 fact-lookup questions, 5 documents, labels
judged by one person. One relevant chunk moves a question's score by 0.33, so
treat the gaps as indicative, not conclusive. Fact-lookup questions favour
BM25; conceptual questions might favour dense.

## Issues Faced

- **Placeholder questions.** The first hybrid run used the default placeholder
  `questions.txt`, so the CSV showed "Replace this line with question N". I
  replaced it with the real questions and reran all three modes.
- **pip installed outside the venv.** Plain `pip` used the global Python. Fixed
  with `python -m pip install ...`.
- **Haystack 3.x moved components out of core.** The extractive reader and the
  Sentence Transformers embedders now come from `transformers-haystack` and
  `sentence-transformers-haystack`. The script imports from the new packages and
  falls back to the old imports on Haystack 2.x.
- **Labelling.** My first labels were filled too quickly. I re-checked every label against the full chunk text, not the 200-character snippet, and the averages changed.
- **Truncated snippets.** The CSV stored only 200 characters per chunk, which
  made labelling hard. Increased to 600 characters.

## Notes

- The PDFs are copyrighted IEEE papers, so they are listed but not committed
  (`pdfs/*.pdf` is in `.gitignore`).
- Environment: Python 3.14, haystack-ai 3.3.0, Windows, per-day `.venv`.

## What's Implemented

- `haystack_pipeline.py` — indexing pipeline plus bm25, dense and hybrid query pipelines
- `evaluate.py` — precision@3 comparison across the three modes
- `questions.txt` — the 10 evaluation questions
- `results/` — `bm25_results.csv`, `dense_results.csv`, `hybrid_results.csv`

## If I Had One More Day

- Tune chunk size and overlap so answers are not split across chunks
- Swap the extractive reader for a generative one to avoid wrong-span answers
- Try a stronger embedding model and report recall as well as precision
- Add more questions so the comparison is less noisy

## Author

Siriyala Nishar
