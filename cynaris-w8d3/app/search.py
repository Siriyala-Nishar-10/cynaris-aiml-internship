"""Haystack search service: BM25 always on, dense retrieval optional.

Documents are plain .txt files in DOCS_DIR (default: sample_docs). They are cleaned,
split into 150-word chunks (20-word overlap) and indexed in memory at start-up.
"""
import glob
import os

from haystack import Document
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class ModeUnavailable(Exception):
    """Raised when a retrieval mode is not enabled on this deployment."""


def load_documents(docs_dir):
    docs = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            docs.append(Document(content=f.read(), meta={"source": os.path.basename(path)}))
    if not docs:
        raise RuntimeError(f"No .txt files found in {docs_dir}")
    return docs


def chunk_documents(docs):
    cleaned = DocumentCleaner().run(documents=docs)["documents"]
    splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=20)
    if hasattr(splitter, "warm_up"):
        splitter.warm_up()
    return splitter.run(documents=cleaned)["documents"]


def build_dense(chunks):
    """Return a function (query, top_k) -> documents. Needs requirements-dense.txt."""
    try:
        from haystack_integrations.components.embedders.sentence_transformers import (
            SentenceTransformersDocumentEmbedder,
            SentenceTransformersTextEmbedder,
        )
    except ImportError:
        try:
            from haystack.components.embedders import (
                SentenceTransformersDocumentEmbedder,
                SentenceTransformersTextEmbedder,
            )
        except ImportError as exc:
            raise RuntimeError("Dense mode needs: pip install -r requirements-dense.txt") from exc
    from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

    doc_embedder = SentenceTransformersDocumentEmbedder(model=EMBED_MODEL)
    doc_embedder.warm_up()
    store = InMemoryDocumentStore()
    store.write_documents(doc_embedder.run(documents=chunks)["documents"])
    retriever = InMemoryEmbeddingRetriever(document_store=store)
    text_embedder = SentenceTransformersTextEmbedder(model=EMBED_MODEL)
    text_embedder.warm_up()

    def search(query, top_k):
        embedding = text_embedder.run(text=query)["embedding"]
        return retriever.run(query_embedding=embedding, top_k=top_k)["documents"]

    return search


class SearchService:
    def __init__(self, docs_dir=None, enable_dense=None):
        docs_dir = docs_dir or os.getenv("DOCS_DIR", "sample_docs")
        if enable_dense is None:
            enable_dense = os.getenv("ENABLE_DENSE", "0") == "1"
        docs = load_documents(docs_dir)
        chunks = chunk_documents(docs)
        self.n_documents, self.n_chunks = len(docs), len(chunks)
        store = InMemoryDocumentStore()
        store.write_documents(chunks)
        self._bm25 = InMemoryBM25Retriever(document_store=store)
        self._dense = build_dense(chunks) if enable_dense else None

    def modes(self):
        return ["bm25"] + (["dense"] if self._dense else [])

    def search(self, query, mode="bm25", top_k=3):
        if mode == "bm25":
            docs = self._bm25.run(query=query, top_k=top_k)["documents"]
        elif mode == "dense":
            if self._dense is None:
                raise ModeUnavailable("dense mode is not enabled (set ENABLE_DENSE=1)")
            docs = self._dense(query, top_k)
        else:
            raise ValueError(f"unknown mode: {mode}")
        return [{"text": d.content, "source": d.meta.get("source", "?"),
                 "score": round(d.score or 0.0, 4)} for d in docs]
