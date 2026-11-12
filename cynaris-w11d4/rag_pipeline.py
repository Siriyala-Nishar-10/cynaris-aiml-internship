import json, os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

def load_documents():
    rows = json.loads((ROOT/"data/knowledge_base.json").read_text(encoding="utf-8"))
    return [Document(page_content=x["content"], metadata={"source":x["id"],"topic":x["topic"]}) for x in rows]

def build_vectorstore(chunk_size=450, chunk_overlap=60):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(load_documents())
    return Chroma.from_documents(
        chunks, OllamaEmbeddings(model=EMBED_MODEL),
        collection_name=f"w11d4_chunks_{chunk_size}",
        persist_directory=str(ROOT/".chroma"/str(chunk_size))
    )

def answer_question(question, vectorstore, k=3):
    docs = vectorstore.as_retriever(search_kwargs={"k":k}).invoke(question)
    contexts = [d.page_content for d in docs]
    prompt = ("Answer using only the context. If the answer is absent, say you do not know. "
              "Do not invent details.\n\nCONTEXT:\n" + "\n\n".join(contexts) +
              "\n\nQUESTION: " + question)
    answer = ChatOllama(model=LLM_MODEL, temperature=0).invoke(prompt).content
    return str(answer).strip(), contexts

def run_rag(qa_pairs, chunk_size=450, k=3):
    store = build_vectorstore(chunk_size)
    output=[]
    for i,pair in enumerate(qa_pairs,1):
        answer,contexts=answer_question(pair["question"],store,k)
        output.append({"question":pair["question"],"answer":answer,
                       "contexts":contexts,"ground_truth":pair["ground_truth"]})
        print(f"[{i}/{len(qa_pairs)}] {pair['question']}\nAnswer: {answer}\n")
    return output
