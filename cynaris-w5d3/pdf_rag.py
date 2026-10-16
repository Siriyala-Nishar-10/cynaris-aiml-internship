"""
W5D3 (Task 3): PDF -> ChromaDB -> top-3 chunks -> Ollama answer
----------------------------------------------------------------
A minimal RAG (retrieval-augmented generation) pipeline:

  1. Extract text from a PDF page by page (keeps page numbers for checking).
  2. Split each page into overlapping chunks.
  3. Embed the chunks with Ollama and store them in ChromaDB (cosine).
  4. Embed the question, retrieve the 3 most similar chunks.
  5. Pass those chunks to the chat model as context and print the answer.

Why overlap? Without it, a sentence cut at a chunk boundary can lose the
meaning needed to match a query.

Run:  python pdf_rag.py [path/to/file.pdf] ["your question"]
Needs: pip install chromadb ollama pypdf
       ollama pull nomic-embed-text && ollama pull llama3.2
"""

import sys

import chromadb
import ollama
from pypdf import PdfReader

DB_PATH = "./chroma_db"
COLLECTION = "pdf_chunks"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
CHUNK_SIZE = 800   # characters
OVERLAP = 150
TOP_K = 3

DEFAULT_PDF = "sample.pdf"
DEFAULT_QUESTION = "What is the main topic of this document?"


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts; returns one vector per text, in order."""
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def chunk_pdf(path: str) -> list[dict]:
    """Return [{'text': ..., 'page': n}, ...] using overlapping windows per page."""
    chunks = []
    step = CHUNK_SIZE - OVERLAP
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        text = " ".join((page.extract_text() or "").split())  # normalize whitespace
        for start in range(0, len(text), step):
            piece = text[start : start + CHUNK_SIZE]
            if piece.strip():
                chunks.append({"text": piece, "page": page_no})
    return chunks


def build_index(pdf_path: str):
    """Embed the PDF chunks and store them in a cosine-distance collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    # start clean so chunks from an earlier PDF don't pollute the results
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    chunks = chunk_pdf(pdf_path)
    if not chunks:
        sys.exit("No text extracted (scanned PDF?). Try a text-based PDF.")

    texts = [c["text"] for c in chunks]
    col.add(
        ids=[f"chunk{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embed(texts),
        metadatas=[{"source": pdf_path, "page": c["page"]} for c in chunks],
    )
    print(f"Chunks: {len(chunks)} stored from {pdf_path}")
    return col


def ask(col, question: str) -> str:
    """Retrieve the top-k chunks and let the LLM answer from them only."""
    res = col.query(query_embeddings=embed([question]), n_results=TOP_K)

    print(f"\nQuestion: {question}")
    print(f"\n=== Top {TOP_K} retrieved chunks ===")
    for rank, (doc, dist, meta) in enumerate(
        zip(res["documents"][0], res["distances"][0], res["metadatas"][0]), start=1
    ):
        print(f"\n[{rank}] sim={1 - dist:.3f} page={meta['page']}\n{doc[:300]}...")

    context = "\n\n---\n\n".join(res["documents"][0])
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say \"I don't know based on the document.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    reply = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},  # deterministic, easier to verify
    )
    return reply["message"]["content"]


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    question = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUESTION

    col = build_index(pdf_path)
    answer = ask(col, question)

    print(f"\n=== Answer ===\n{answer}")
    print("\nVERIFY: open the PDF at the page(s) shown above and confirm the answer matches")


if __name__ == "__main__":
    main()