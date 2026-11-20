"""RAG backends behind one small interface, so the API can be tested without Ollama."""
import os
from dataclasses import dataclass, field


@dataclass
class RagResult:
    answer: str
    sources: list = field(default_factory=list)


class StubBackend:
    """Canned answers. Used for CI tests and for running the container without Ollama."""

    def query(self, question: str) -> RagResult:
        return RagResult(answer=f"[stub] You asked: {question}", sources=["stub.txt"])


class LlamaIndexBackend:
    """LlamaIndex over the .txt files in DATA_DIR, with local Ollama models.

    Environment variables: OLLAMA_HOST, LLM_MODEL, EMBED_MODEL, DATA_DIR, TOP_K.
    The imports are inside __init__ so tests never need LlamaIndex installed.
    """

    def __init__(self):
        from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama

        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        Settings.llm = Ollama(model=os.getenv("LLM_MODEL", "llama3.2"), base_url=host,
                              request_timeout=180.0)
        embed_name = os.getenv("EMBED_MODEL", "nomic-embed-text")
        Settings.embed_model = OllamaEmbedding(model_name=embed_name, base_url=host)
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        data_dir = os.getenv("DATA_DIR", "data")
        docs = SimpleDirectoryReader(data_dir, required_exts=[".txt"]).load_data()
        if not docs:
            raise RuntimeError(f"No .txt files found in {data_dir}")
        index = VectorStoreIndex.from_documents(docs)
        self.engine = index.as_query_engine(similarity_top_k=int(os.getenv("TOP_K", "3")))

    def query(self, question: str) -> RagResult:
        resp = self.engine.query(question)
        sources = sorted({os.path.basename(n.node.metadata.get("file_name", "?"))
                          for n in resp.source_nodes})
        return RagResult(answer=str(resp).strip(), sources=sources)


def backend_name() -> str:
    return os.getenv("RAG_BACKEND", "llamaindex")


def make_backend():
    name = backend_name()
    if name == "stub":
        return StubBackend()
    if name == "llamaindex":
        return LlamaIndexBackend()
    raise ValueError(f"Unknown RAG_BACKEND: {name}")
