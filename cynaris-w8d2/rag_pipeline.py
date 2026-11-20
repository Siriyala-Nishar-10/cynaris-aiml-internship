"""RAG pipeline: LangChain + ChromaDB + Ollama.

chunk_size is measured in characters (RecursiveCharacterTextSplitter default).
The Chroma store is in-memory and rebuilt for every configuration, so a new
chunk size or k never reuses stale vectors.
"""
import glob
import os
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below. If the context does not contain "
    "the answer, say you do not know. Be concise.\n\nContext:\n{context}\n\n"
    "Question: {question}\nAnswer:")


def load_documents(data_dir="data"):
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            docs.append(Document(page_content=f.read(), metadata={"source": os.path.basename(path)}))
    if not docs:
        raise SystemExit(f"No .txt files in ./{data_dir}. Run pdf_to_txt.py first.")
    return docs


class RagPipeline:
    def __init__(self, chunk_size, k, data_dir="data"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size // 10)
        chunks = splitter.split_documents(load_documents(data_dir))
        embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
        self.store = Chroma.from_documents(chunks, embeddings,
                                           collection_name=f"w8d2_{uuid.uuid4().hex[:8]}")
        self.retriever = self.store.as_retriever(search_kwargs={"k": k})
        self.llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST, temperature=0)
        self.n_chunks = len(chunks)

    def ask(self, question):
        """Return (answer, list of retrieved context strings)."""
        contexts = [d.page_content for d in self.retriever.invoke(question)]
        prompt = PROMPT.format(context="\n\n".join(contexts), question=question)
        return self.llm.invoke(prompt).content.strip(), contexts
