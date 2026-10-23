"""W7D2 - Haystack retrieval: BM25 vs Dense vs Hybrid (reciprocal rank fusion).

Usage:
    python haystack_pipeline.py bm25
    python haystack_pipeline.py dense
    python haystack_pipeline.py hybrid   # BM25 + dense merged with DocumentJoiner (RRF)

Put 5 PDFs in ./pdfs and 10 questions (one per line) in questions.txt.
Each run writes results/<mode>_results.csv with the top-3 chunks per question.
Fill the `relevant` column by hand (1/0), then run `python evaluate.py`.
"""
import csv, glob, os, sys

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Haystack 3.x moved these into integration packages
# (pip install sentence-transformers-haystack transformers-haystack);
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
TOP_K = 3                     # chunks judged for precision@3
CANDIDATES = 5                # chunks each retriever hands to the joiner in hybrid mode
PDF_DIR, QUESTIONS, OUT_DIR = "pdfs", "questions.txt", "results"
MODES = ("bm25", "dense", "hybrid")


def build_index(mode):
    """Converter -> Cleaner -> Splitter -> (Embedder) -> DocumentStore."""
    store = InMemoryDocumentStore()
    idx = Pipeline()
    idx.add_component("converter", PyPDFToDocument())
    idx.add_component("cleaner", DocumentCleaner())
    idx.add_component("splitter", DocumentSplitter(split_by="word", split_length=150, split_overlap=20))
    idx.add_component("writer", DocumentWriter(document_store=store))
    idx.connect("converter.documents", "cleaner.documents")
    idx.connect("cleaner.documents", "splitter.documents")
    if mode in ("dense", "hybrid"):                      # BM25 alone needs no embeddings
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


def build_query_pipeline(mode, store):
    """Retriever(s) -> Reader. Only the retrieval part changes between modes."""
    p = Pipeline()
    if mode == "bm25":
        p.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=TOP_K))
        last = "retriever"
    elif mode == "dense":
        p.add_component("text_embedder", SentenceTransformersTextEmbedder(model=EMBED_MODEL))
        p.add_component("retriever", InMemoryEmbeddingRetriever(document_store=store, top_k=TOP_K))
        p.connect("text_embedder.embedding", "retriever.query_embedding")
        last = "retriever"
    else:  # hybrid
        p.add_component("text_embedder", SentenceTransformersTextEmbedder(model=EMBED_MODEL))
        p.add_component("bm25", InMemoryBM25Retriever(document_store=store, top_k=CANDIDATES))
        p.add_component("dense", InMemoryEmbeddingRetriever(document_store=store, top_k=CANDIDATES))
        p.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=TOP_K))
        p.connect("text_embedder.embedding", "dense.query_embedding")
        p.connect("bm25.documents", "joiner.documents")
        p.connect("dense.documents", "joiner.documents")
        last = "joiner"
    p.add_component("reader", ExtractiveReader())
    p.connect(f"{last}.documents", "reader.documents")
    return p, last


def ask(p, last, mode, question):
    data = {"reader": {"query": question, "top_k": 1}}
    if mode == "bm25":
        data["retriever"] = {"query": question}
    elif mode == "dense":
        data["text_embedder"] = {"text": question}
    else:
        data["text_embedder"] = {"text": question}
        data["bm25"] = {"query": question}
    out = p.run(data, include_outputs_from={last})
    docs = out[last]["documents"]
    answers = [a for a in out["reader"]["answers"] if a.data]   # drop the empty "no answer" entry
    return docs, (answers[0].data if answers else "")


def main(mode):
    questions = [q.strip() for q in open(QUESTIONS, encoding="utf-8") if q.strip()]
    store = build_index(mode)
    pipe, last = build_query_pipeline(mode, store)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{mode}_results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["q_id", "question", "rank", "source", "score", "snippet", "reader_answer", "relevant"])
        for i, q in enumerate(questions, 1):
            docs, answer = ask(pipe, last, mode, q)
            for r, d in enumerate(docs[:TOP_K], 1):
                w.writerow([i, q, r, os.path.basename(d.meta.get("file_path", "?")),
                            round(d.score or 0, 3), d.content[:200].replace("\n", " "),
                            answer if r == 1 else "", ""])
            print(f"Q{i}: {q}\n   -> {answer or '(no extractive answer)'}")
    print(f"\nSaved {path}. Fill the 'relevant' column (1/0), then run evaluate.py")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in MODES:
        sys.exit("Usage: python haystack_pipeline.py [bm25|dense|hybrid]")
    main(sys.argv[1])
