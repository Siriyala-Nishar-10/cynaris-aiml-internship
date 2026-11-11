"""W7D5 - Multi-document RAG with LlamaIndex + Ollama.

Two designs over the same 5 papers and the same 10 questions:
    flat    one VectorStoreIndex over all documents (the W7D3 design)
    router  one index per document; an LLM router picks which document to query

Usage:
    python multi_doc_rag.py flat
    python multi_doc_rag.py router

Needs Ollama running (ollama pull nomic-embed-text, ollama pull llama3.2).
Override the chat model with  $env:LLM_MODEL="qwen2.5:3b".
Writes results/<mode>_results.csv and results/<mode>_meta.json.
"""
import csv, glob, json, os, sys, time

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from utils import DESCRIPTIONS, EXPECTED_DOC, doc_key, keyword_hit

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DATA_DIR, QUESTIONS, OUT_DIR = "data", "questions.txt", "results"
TOP_K = 3

Settings.llm = Ollama(model=LLM_MODEL, request_timeout=180.0)
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
Settings.chunk_size = 512
Settings.chunk_overlap = 50


def load_documents():
    """Return {file_path: [Document, ...]} for every .txt file in ./data."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not files:
        sys.exit(f"No .txt files in ./{DATA_DIR}. Run pdf_to_txt.py first.")
    return {f: SimpleDirectoryReader(input_files=[f]).load_data() for f in files}


def build_engine(mode, per_file):
    if mode == "flat":
        all_docs = [d for docs in per_file.values() for d in docs]
        return VectorStoreIndex.from_documents(all_docs).as_query_engine(similarity_top_k=TOP_K)
    tools = []
    for path, docs in per_file.items():
        key = doc_key(os.path.basename(path)) or os.path.basename(path)
        engine = VectorStoreIndex.from_documents(docs).as_query_engine(similarity_top_k=TOP_K)
        tools.append(QueryEngineTool.from_defaults(
            query_engine=engine, name=key,
            description=DESCRIPTIONS.get(key, f"Content of {os.path.basename(path)}")))
    return RouterQueryEngine(selector=LLMSingleSelector.from_defaults(), query_engine_tools=tools)


def main(mode):
    questions = [q.strip() for q in open(QUESTIONS, encoding="utf-8") if q.strip()]
    per_file = load_documents()
    t = time.perf_counter()
    engine = build_engine(mode, per_file)
    build_s = time.perf_counter() - t
    print(f"[{mode}] built over {len(per_file)} documents in {build_s:.1f}s")

    try:
        engine.query("warm up")                   # load models once so question 1 is not penalised
    except Exception:
        pass

    os.makedirs(OUT_DIR, exist_ok=True)
    errors = 0
    with open(os.path.join(OUT_DIR, f"{mode}_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["q_id", "question", "answer", "total_s", "sources", "top_doc_ok", "keyword_hit", "correct"])
        for i, q in enumerate(questions, 1):
            t0 = time.perf_counter()
            try:
                resp = engine.query(q)
                answer = str(resp).strip().replace("\n", " ")
                files = [os.path.basename(n.node.metadata.get("file_name", "?")) for n in resp.source_nodes]
            except Exception as e:                # 3B routers sometimes return malformed selections
                answer, files, errors = f"ERROR: {type(e).__name__}: {e}"[:200], [], errors + 1
            total_s = time.perf_counter() - t0
            top_doc_ok = bool(files) and doc_key(files[0]) == EXPECTED_DOC[i]
            hit = keyword_hit(i, answer)
            w.writerow([i, q, answer, round(total_s, 2), "; ".join(sorted(set(files))), int(top_doc_ok), int(hit), ""])
            print(f"Q{i}: {q}\n   -> {answer[:150]}\n   {total_s:.1f}s | right doc: {top_doc_ok} | keyword hit: {hit}")
    with open(os.path.join(OUT_DIR, f"{mode}_meta.json"), "w") as f:
        json.dump({"mode": mode, "build_s": round(build_s, 2), "errors": errors, "documents": len(per_file),
                   "llm": LLM_MODEL, "embed": EMBED_MODEL, "top_k": TOP_K}, f, indent=2)
    print(f"\nSaved results/{mode}_results.csv. Verify each answer against the paper, fill `correct` (1/0), then run summarise.py")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("flat", "router"):
        sys.exit("Usage: python multi_doc_rag.py [flat|router]")
    main(sys.argv[1])
