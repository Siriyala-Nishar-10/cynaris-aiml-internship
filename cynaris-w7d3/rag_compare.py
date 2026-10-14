"""W7D3 - LlamaIndex document indexing and querying: in-memory index vs ChromaDB.

Usage:
    python rag_compare.py memory     # VectorStoreIndex, default in-memory vector store
    python rag_compare.py chroma     # same index, ChromaDB as the vector store

Needs Ollama running with an embedding model and a chat model, e.g.
    ollama pull nomic-embed-text
    ollama pull llama3.2
Override with env vars:  $env:LLM_MODEL="qwen2.5:3b"   $env:EMBED_MODEL="nomic-embed-text"

Each run writes results/<mode>_results.csv (answer, latencies, source files) and
results/<mode>_meta.json (index build time). Fill the `correct` column by hand
(1/0) after checking each answer against the source text, then run summarise.py.
"""
import csv, json, os, sys, time

import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DATA_DIR, QUESTIONS, OUT_DIR, CHROMA_DIR = "data", "questions.txt", "results", "chroma_db"
TOP_K = 3

Settings.llm = Ollama(model=LLM_MODEL, request_timeout=180.0)
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
Settings.chunk_size = 512
Settings.chunk_overlap = 50


def build_index(mode):
    docs = SimpleDirectoryReader(DATA_DIR, required_exts=[".txt"]).load_data()
    if not docs:
        sys.exit(f"No .txt files in ./{DATA_DIR}. Run pdf_to_txt.py first.")
    t = time.perf_counter()
    if mode == "memory":
        index = VectorStoreIndex.from_documents(docs)
    else:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection("w7d3")          # rebuild from scratch so build time is comparable
        except Exception:
            pass
        collection = client.create_collection("w7d3")
        storage = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
        index = VectorStoreIndex.from_documents(docs, storage_context=storage)
    build_s = time.perf_counter() - t
    print(f"[{mode}] indexed {len(docs)} documents in {build_s:.1f}s")
    return index, build_s, len(docs)


def main(mode):
    questions = [q.strip() for q in open(QUESTIONS, encoding="utf-8") if q.strip()]
    index, build_s, n_docs = build_index(mode)
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    engine = index.as_query_engine(similarity_top_k=TOP_K)

    retriever.retrieve("warm up")                      # load models once so the first question is not penalised
    engine.query("warm up")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{mode}_results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["q_id", "question", "answer", "retrieval_s", "total_s", "sources", "top_score", "correct"])
        for i, q in enumerate(questions, 1):
            t0 = time.perf_counter(); retriever.retrieve(q); retrieval_s = time.perf_counter() - t0
            t1 = time.perf_counter(); resp = engine.query(q); total_s = time.perf_counter() - t1
            nodes = resp.source_nodes
            sources = "; ".join(sorted({os.path.basename(n.node.metadata.get("file_name", "?")) for n in nodes}))
            top = round(nodes[0].score, 3) if nodes and nodes[0].score is not None else ""
            answer = str(resp).strip().replace("\n", " ")
            w.writerow([i, q, answer, round(retrieval_s, 3), round(total_s, 2), sources, top, ""])
            print(f"Q{i}: {q}\n   -> {answer[:160]}\n   retrieval {retrieval_s:.2f}s | total {total_s:.1f}s | {sources}")
    with open(os.path.join(OUT_DIR, f"{mode}_meta.json"), "w") as f:
        json.dump({"mode": mode, "build_s": round(build_s, 2), "documents": n_docs,
                   "llm": LLM_MODEL, "embed": EMBED_MODEL, "top_k": TOP_K}, f, indent=2)
    print(f"\nSaved {path}. Check each answer against the source text, fill `correct` (1/0), then run summarise.py")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("memory", "chroma"):
        sys.exit("Usage: python rag_compare.py [memory|chroma]")
    main(sys.argv[1])
