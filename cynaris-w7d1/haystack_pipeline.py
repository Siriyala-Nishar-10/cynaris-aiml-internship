"""W7D1 - Haystack pipeline: PDF converter -> DocumentStore -> Retriever -> Reader.

Usage:
    python haystack_pipeline.py bm25     # keyword retrieval
    python haystack_pipeline.py dense    # embedding retrieval (same 10 questions)

Put 5 PDFs in ./pdfs and your 10 questions (one per line) in questions.txt.
Each run writes results/<mode>_results.csv with the top-3 retrieved chunks per
question. Fill the `relevant` column by hand (1 = chunk answers the question,
0 = not), then run `python evaluate.py`.
"""
import csv, glob, os, sys

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Haystack 3.x moved these components into integration packages
# (pip install sentence-transformers-haystack transformers-haystack).
# Haystack 2.x still has them in core, so fall back to the old imports.
try:
    from haystack_integrations.components.embedders.sentence_transformers import (
        SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder)
    from haystack_integrations.components.readers.transformers import (
        TransformersExtractiveReader as ExtractiveReader)
except ImportError:
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
    from haystack.components.readers import ExtractiveReader

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3          # chunks we judge for precision@3
PDF_DIR, QUESTIONS, OUT_DIR = "pdfs", "questions.txt", "results"


def build_index(mode: str) -> InMemoryDocumentStore:
    """FileConverter -> Cleaner -> Splitter -> (Embedder) -> DocumentStore."""
    store = InMemoryDocumentStore()
    idx = Pipeline()
    idx.add_component("converter", PyPDFToDocument())
    idx.add_component("cleaner", DocumentCleaner())
    idx.add_component("splitter", DocumentSplitter(split_by="word", split_length=150, split_overlap=20))
    idx.add_component("writer", DocumentWriter(document_store=store))
    idx.connect("converter.documents", "cleaner.documents")
    idx.connect("cleaner.documents", "splitter.documents")
    if mode == "dense":
        idx.add_component("embedder", SentenceTransformersDocumentEmbedder(model=EMBED_MODEL))
        idx.connect("splitter.documents", "embedder.documents")
        idx.connect("embedder.documents", "writer.documents")
    else:
        idx.connect("splitter.documents", "writer.documents")

    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdfs:
        sys.exit(f"No PDFs found in ./{PDF_DIR}. Add your 5 PDFs first.")
    idx.run({"converter": {"sources": pdfs}})
    print(f"[{mode}] indexed {len(pdfs)} PDFs -> {store.count_documents()} chunks")
    return store


def build_query_pipeline(mode: str, store: InMemoryDocumentStore) -> Pipeline:
    """Retriever -> Reader. Only the retriever differs between the two modes."""
    p = Pipeline()
    if mode == "dense":
        p.add_component("text_embedder", SentenceTransformersTextEmbedder(model=EMBED_MODEL))
        p.add_component("retriever", InMemoryEmbeddingRetriever(document_store=store, top_k=TOP_K))
        p.connect("text_embedder.embedding", "retriever.query_embedding")
    else:
        p.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=TOP_K))
    p.add_component("reader", ExtractiveReader())          # extractive QA over the retrieved chunks
    p.connect("retriever.documents", "reader.documents")
    return p


def ask(p: Pipeline, mode: str, question: str):
    data = {"reader": {"query": question, "top_k": 1}}
    data["text_embedder" if mode == "dense" else "retriever"] = (
        {"text": question} if mode == "dense" else {"query": question})
    out = p.run(data, include_outputs_from={"retriever"})
    docs = out["retriever"]["documents"]
    answers = [a for a in out["reader"]["answers"] if a.data]   # drop the empty "no answer" entry
    return docs, (answers[0].data if answers else "")


def main(mode: str):
    questions = [q.strip() for q in open(QUESTIONS, encoding="utf-8") if q.strip()]
    store = build_index(mode)
    pipe = build_query_pipeline(mode, store)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{mode}_results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["q_id", "question", "rank", "source", "score", "snippet", "reader_answer", "relevant"])
        for i, q in enumerate(questions, 1):
            docs, answer = ask(pipe, mode, q)
            for r, d in enumerate(docs, 1):
                w.writerow([i, q, r, os.path.basename(d.meta.get("file_path", "?")),
                            round(d.score or 0, 3), d.content[:200].replace("\n", " "),
                            answer if r == 1 else "", ""])
            print(f"Q{i}: {q}\n   -> {answer or '(no extractive answer)'}")
    print(f"\nSaved {path}. Fill the 'relevant' column (1/0), then run evaluate.py")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("bm25", "dense"):
        sys.exit("Usage: python haystack_pipeline.py [bm25|dense]")
    main(sys.argv[1])