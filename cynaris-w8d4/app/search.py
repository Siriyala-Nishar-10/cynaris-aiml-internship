"""Haystack search service: BM25 always on, dense retrieval optional.

Documents are plain .txt files. They are cleaned, split into word chunks and
indexed in memory when the service starts.
"""
from pathlib import Path

from haystack import Document
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from app.config import Settings, load_settings


class ModeUnavailable(Exception):
    """Raised when a retrieval mode is not enabled on this deployment."""


class NoDocumentsError(RuntimeError):
    """Raised when the documents folder holds no .txt files."""


def load_documents(docs_dir):
    """Read every .txt file in `docs_dir` into a Document whose meta holds the file name."""
    docs = []
    for path in sorted(Path(docs_dir).glob("*.txt")):
        docs.append(Document(content=path.read_text(encoding="utf-8"),
                             meta={"source": path.name}))
    if not docs:
        raise NoDocumentsError(f"No .txt files found in {docs_dir}")
    return docs


def chunk_documents(docs, words=150, overlap=20):
    """Clean the documents and split them into chunks of at most `words` words."""
    cleaned = DocumentCleaner().run(documents=docs)["documents"]
    splitter = DocumentSplitter(split_by="word", split_length=words, split_overlap=overlap)
    if hasattr(splitter, "warm_up"):
        splitter.warm_up()
    return splitter.run(documents=cleaned)["documents"]


def build_dense(chunks, embed_model):
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

    doc_embedder = SentenceTransformersDocumentEmbedder(model=embed_model)
    doc_embedder.warm_up()
    store = InMemoryDocumentStore()
    store.write_documents(doc_embedder.run(documents=chunks)["documents"])
    retriever = InMemoryEmbeddingRetriever(document_store=store)
    text_embedder = SentenceTransformersTextEmbedder(model=embed_model)
    text_embedder.warm_up()

    def search(query, top_k):
        embedding = text_embedder.run(text=query)["embedding"]
        return retriever.run(query_embedding=embedding, top_k=top_k)["documents"]

    return search


class SearchService:
    """Indexes the documents once and answers search requests."""

    def __init__(self, settings: Settings | None = None):
        settings = settings or load_settings()
        docs = load_documents(settings.docs_dir)
        chunks = chunk_documents(docs, settings.chunk_words, settings.chunk_overlap)
        self.n_documents, self.n_chunks = len(docs), len(chunks)
        store = InMemoryDocumentStore()
        store.write_documents(chunks)
        self._bm25 = InMemoryBM25Retriever(document_store=store)
        self._dense = build_dense(chunks, settings.embed_model) if settings.enable_dense else None

    def modes(self):
        return ["bm25"] + (["dense"] if self._dense else [])

    def search(self, query, mode="bm25", top_k=3):
        """Return up to `top_k` hits as dicts with text, source and score (best first)."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
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
