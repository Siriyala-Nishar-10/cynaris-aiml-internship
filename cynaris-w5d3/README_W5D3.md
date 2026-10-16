# W5D3: ChromaDB, Vector Store Setup & Embedding Documents

Hands-on work for Week 5 Day 3: building a local vector store with ChromaDB, embedding documents with Ollama, running similarity searches, and using retrieved PDF chunks to ground an LLM answer (a basic RAG pipeline).

## What's in this repo

| File               | Purpose                                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------------------- |
| `ingest_docs.py`   | Creates a cosine-distance collection, embeds 20 documents, runs similarity search and metadata filtering |
| `pdf_rag.py`       | Chunks a PDF, embeds and stores the chunks, retrieves the top 3 and asks the LLM                         |
| `sample.pdf`       | PDF used for the retrieval demo                                                                          |
| `evidence/`        | Screenshots of outputs and verification                                                                  |
| `requirements.txt` | Python dependencies                                                                                      |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ollama pull nomic-embed-text     # embedding model
ollama pull llama3.2             # chat model
```

Ollama must be running locally (`ollama serve` if it isn't already).

## Usage

```bash
python ingest_docs.py    # Tasks 1 and 2
python pdf_rag.py        # Task 3
```

## How it works

1. **Collection:** created with `metadata={"hnsw:space": "cosine"}`. Chroma defaults to L2 distance, so cosine has to be set at creation.
2. **Embedding:** each text is embedded with `nomic-embed-text` via Ollama and stored alongside its document and metadata (`category`).
3. **Similarity search:** the query is embedded the same way. Chroma returns cosine _distance_, so similarity = `1 - distance`.
4. **Metadata filtering:** `where={"category": "ml"}` restricts the search to matching documents.
5. **PDF retrieval:** text is split into [800]-character chunks with [150] overlap, embedded, and stored. The top 3 chunks for a question are passed to `llama3.2` as context, with an instruction to answer only from that context.

## Results and verification

- Collection count: 20 documents
- Query `"How do models learn from data?"` returned: "Overfitting happens when a model memorizes training data instead of generalizing" (sim=0.680), followed by "Neural networks learn by adjusting weights through backpropagation" (sim=0.606). Both are ML documents about model training, so they match the query. Rank 3 (MLflow, sim=0.576) is a weaker match on the word "model".
- With the `category = geography` filter, all 3 results were geography documents (sim 0.37 to 0.41), much lower than the unfiltered matches (0.58 to 0.68). This confirms the filter works and excludes the relevant ML documents.
- PDF question `"What is the main topic of this document?"` was answered correctly: the document is about an intelligent banking assistant that uses RAG over internal documentation to streamline HR workload. The top-3 chunks came from pages 2 and 5, and I checked the answer against page 2 (Problem Formulation) of the PDF.

## Notes and trade-offs

- Chunk size and overlap affect retrieval quality. Overlap prevents sentences being cut off at chunk boundaries.
- ChromaDB is simple to set up and supports persistence and metadata filtering. FAISS is faster at very large scale but needs you to manage storage and metadata yourself.
- Re-running ingestion can cause duplicate ID errors; the scripts use `[upsert / delete chroma_db]` to avoid this.

## Self-review checklist

- [x] Code runs from a clean environment
- [x] Collection uses cosine distance
- [x] 20 documents embedded, with metadata
- [x] Similarity search and metadata filter verified manually
- [x] PDF retrieval returns sensible top-3 chunks
- [x] LLM answer checked against the source PDF
- [x] `chroma_db/` is in `.gitignore`

## Author

Siriyala Nishar
