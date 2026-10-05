"""
W5D5: Local Q&A bot with Ollama + ChromaDB (Week 5 project)
-------------------------------------------------------------
A fully local question-answering bot over your own documents. Nothing leaves
the machine: embeddings, retrieval and generation all run through Ollama and
a persistent ChromaDB store.

Pipeline:

  1. Ingest: read every .pdf / .txt / .md file in the docs folder, split the
     text into overlapping chunks (PDF chunks keep their page number).
  2. Index: embed the chunks with Ollama and store them in a cosine ChromaDB
     collection, with `source` and `page` metadata for citations.
  3. Retrieve: embed the question with the SAME model, fetch the top-k chunks
     and keep only those above a similarity threshold.
  4. Answer: send the chunks plus a custom system prompt to the chat model,
     which must answer ONLY from the context and cite its sources.
  5. Guardrail: if no chunk is similar enough, the bot says it doesn't know
     without calling the LLM at all (cheaper and avoids hallucination).

Chroma returns cosine DISTANCE, so similarity = 1 - distance.

Run:  python qa_bot.py                          interactive chat
      python qa_bot.py -q "your question"       one-shot answer
      python qa_bot.py --model qwen2.5:3b       use a different chat model
      python qa_bot.py --rebuild                re-index the docs folder
Needs: pip install chromadb ollama pypdf
       ollama pull nomic-embed-text && ollama pull llama3.2:3b
       a docs/ folder containing at least one PDF, TXT or MD file
"""

import argparse
import sys
import time
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader

DOCS_DIR = "docs"
DB_PATH = "./chroma_db"
COLLECTION = "qa_bot"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"

CHUNK_SIZE = 800    # characters
OVERLAP = 150
TOP_K = 3
MIN_SIM = 0.40      # below this, the chunk is treated as irrelevant
EMBED_BATCH = 32

NO_ANSWER = "I don't know based on the documents."

SYSTEM_PROMPT = (
    "You are a careful document Q&A assistant running locally.\n"
    "Rules:\n"
    "1. Answer ONLY from the provided context. Do not use outside knowledge.\n"
    f"2. If the context does not contain the answer, reply exactly: {NO_ANSWER}\n"
    "3. Be concise: at most 4 sentences.\n"
    "4. Cite the sources you used in square brackets, like [file.pdf p.2]."
)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts in small batches; returns one vector per text, in order."""
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        vectors.extend(ollama.embed(model=EMBED_MODEL, input=batch)["embeddings"])
    return vectors


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping windows of CHUNK_SIZE characters."""
    text = " ".join(text.split())  # normalize whitespace
    step = CHUNK_SIZE - OVERLAP
    return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), step) if text[i : i + CHUNK_SIZE].strip()]


def load_chunks(docs_dir: str) -> list[dict]:
    """Read all supported files; return [{'text', 'source', 'page'}, ...]."""
    folder = Path(docs_dir)
    if not folder.is_dir():
        sys.exit(f"Docs folder '{docs_dir}' not found. Create it and add a PDF, TXT or MD file.")

    chunks = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() == ".pdf":
            pages = [(n, p.extract_text() or "") for n, p in enumerate(PdfReader(str(path)).pages, start=1)]
        elif path.suffix.lower() in {".txt", ".md"}:
            pages = [(1, path.read_text(encoding="utf-8", errors="ignore"))]
        else:
            continue
        for page_no, text in pages:
            for piece in chunk_text(text):
                chunks.append({"text": piece, "source": path.name, "page": page_no})

    if not chunks:
        sys.exit(f"No text found in '{docs_dir}' (scanned PDF?). Add a text-based PDF, TXT or MD file.")
    return chunks


def build_index(docs_dir: str = DOCS_DIR, rebuild: bool = False):
    """Create the cosine collection and (re)load it from the docs folder."""
    client = chromadb.PersistentClient(path=DB_PATH)
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    # Index only when empty, or when --rebuild was requested (embedding is the slow part)
    if col.count() == 0:
        chunks = load_chunks(docs_dir)
        texts = [c["text"] for c in chunks]
        col.upsert(
            ids=[f"{c['source']}-p{c['page']}-{i}" for i, c in enumerate(chunks)],
            documents=texts,
            embeddings=embed(texts),
            metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
        )
        print(f"Indexed {len(chunks)} chunks from '{docs_dir}'")
    else:
        print(f"Using existing index: {col.count()} chunks (use --rebuild to refresh)")
    return col


def retrieve(col, question: str, k: int = TOP_K) -> list[dict]:
    """Return the k nearest chunks with cosine similarity, best first."""
    res = col.query(query_embeddings=embed([question]), n_results=k)
    return [
        {"text": doc, "sim": 1 - dist, "source": meta["source"], "page": meta["page"]}
        for doc, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0])
    ]


def answer(col, question: str, model: str = CHAT_MODEL, k: int = TOP_K, min_sim: float = MIN_SIM) -> dict:
    """Retrieve, apply the similarity guardrail, then ask the LLM.

    Returns a dict with the answer, the sources used, timing and token count.
    Set min_sim=0 to always call the LLM (used when comparing models).
    """
    start = time.perf_counter()
    hits = retrieve(col, question, k)
    relevant = [h for h in hits if h["sim"] >= min_sim]

    result = {"question": question, "model": model, "sources": hits, "llm_called": False,
              "eval_tokens": None, "error": None}

    if not relevant:  # guardrail: nothing close enough, don't risk a made-up answer
        result["answer"] = NO_ANSWER
    else:
        context = "\n\n".join(f"[{h['source']} p.{h['page']}] {h['text']}" for h in relevant)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        try:
            reply = ollama.chat(model=model, messages=messages, options={"temperature": 0})
            result["answer"] = reply["message"]["content"].strip()
            result["eval_tokens"] = getattr(reply, "eval_count", None)
            result["llm_called"] = True
        except Exception as exc:  # model not pulled, server down, etc.
            result["answer"] = f"ERROR: {exc}"
            result["error"] = str(exc)

    result["seconds"] = round(time.perf_counter() - start, 2)
    return result


def print_result(r: dict) -> None:
    """Show the answer, then the retrieved sources with similarity scores."""
    print(f"\nAnswer ({r['model']}, {r['seconds']}s):\n{r['answer']}")
    print("\nSources retrieved:")
    for h in r["sources"]:
        print(f"  sim={h['sim']:.3f}  {h['source']} p.{h['page']}")
    if not r["llm_called"] and not r["error"]:
        print("  (no chunk above the similarity threshold, LLM not called)")


def chat_loop(col, model: str, k: int, min_sim: float) -> None:
    """Simple interactive loop; type 'exit' to quit."""
    print(f"\nLocal Q&A bot | model={model} | top-{k} | min similarity={min_sim}")
    print("Type a question, or 'exit' to quit.")
    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            print_result(answer(col, q, model, k, min_sim))


def main() -> None:
    p = argparse.ArgumentParser(description="Local Q&A bot (Ollama + ChromaDB)")
    p.add_argument("-q", "--question", help="ask one question and exit")
    p.add_argument("--model", default=CHAT_MODEL, help=f"chat model (default {CHAT_MODEL})")
    p.add_argument("--docs", default=DOCS_DIR, help=f"documents folder (default {DOCS_DIR})")
    p.add_argument("--rebuild", action="store_true", help="delete and rebuild the index")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--min-sim", type=float, default=MIN_SIM)
    args = p.parse_args()

    col = build_index(args.docs, args.rebuild)
    if args.question:
        print(f"\nQuestion: {args.question}")
        print_result(answer(col, args.question, args.model, args.top_k, args.min_sim))
    else:
        chat_loop(col, args.model, args.top_k, args.min_sim)


if __name__ == "__main__":
    main()
