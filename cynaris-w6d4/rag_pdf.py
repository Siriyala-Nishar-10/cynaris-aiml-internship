"""W6D4 Task 3: embed a PDF, retrieve top-3 chunks, answer with Ollama.

Prereqs:
    pip install -r requirements.txt
    ollama pull nomic-embed-text
    ollama pull llama3.2

Usage:
    python rag_pdf.py sample.pdf "Your question here"
    python rag_pdf.py sample.pdf            # prompts for the question
"""
import argparse

import chromadb
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

COLLECTION = "pdf_rag"
TOP_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

emb = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0)
client = chromadb.PersistentClient(path="./chroma_db")


def make_collection(name: str):
    """Create a fresh cosine-space collection (works across Chroma versions)."""
    try:
        client.delete_collection(name)
    except Exception:
        pass
    try:
        return client.create_collection(
            name, configuration={"hnsw": {"space": "cosine"}}
        )
    except Exception:
        return client.create_collection(name, metadata={"hnsw:space": "cosine"})


def ingest(pdf_path: str):
    """Load PDF -> split into chunks -> embed -> store in Chroma."""
    reader = PdfReader(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    # Split page by page so every chunk keeps its (1-indexed) page number.
    texts, metas = [], []
    for page_no, page in enumerate(reader.pages, start=1):
        for chunk in splitter.split_text(page.extract_text() or ""):
            texts.append(chunk)
            metas.append({"page": page_no})

    if not texts:
        raise SystemExit("No text extracted. Is the PDF scanned (image only)?")

    col = make_collection(COLLECTION)
    col.add(
        ids=[f"chunk{i}" for i in range(len(texts))],
        documents=texts,
        embeddings=emb.embed_documents(texts),
        metadatas=metas,
    )
    print(f"Ingested {len(reader.pages)} pages into {len(texts)} chunks.")
    return col


def retrieve(col, question: str, k: int = TOP_K):
    res = col.query(query_embeddings=[emb.embed_query(question)], n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def answer(col, question: str) -> str:
    hits = retrieve(col, question)

    print("\nRetrieved chunks:")
    for text, meta, dist in hits:
        print(f"  [page {meta['page']}] sim={1 - dist:.3f} :: {text[:120]!r}...")

    context = "\n\n".join(f"(page {m['page']}) {t}" for t, m, _ in hits)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say "
        "\"I don't know based on the document.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return llm.invoke(prompt).content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", default="sample.pdf")
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()

    question = args.question or input("Question about the PDF: ")
    collection = ingest(args.pdf)

    print(f"\nQ: {question}")
    print(f"A: {answer(collection, question)}")

    # Hallucination check: a question the PDF should NOT cover.
    off_topic = "What is the capital of Australia?"
    print(f"\nQ (off-topic check): {off_topic}")
    print(f"A: {answer(collection, off_topic)}")