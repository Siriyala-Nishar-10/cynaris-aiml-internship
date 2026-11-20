"""W6D5 Week 6 Project: Document Chatbot with LangChain (Tasks 1 & 2).

Task 1: PromptTemplate -> Ollama LLM -> OutputParser chain (with PDF retrieval),
        tested on 5 questions.
Task 2: the same bot with conversation buffer memory, verified over 5 turns.

Prereqs:
    pip install -r requirements.txt
    ollama pull nomic-embed-text
    ollama pull llama3.2

Usage:
    python doc_chatbot.py sample.pdf
"""
import argparse
import os

import chromadb
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")  # e.g. llama3.2:1b if memory is tight
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "w6d5_doc"
TOP_K = 5

# ---- EDIT THESE to match the PDF you use (defaults suit the HR RAG paper) ----
TASK1_QUESTIONS = [
    "What is the purpose of the RAG system described in this paper?",
    "What is the input data of the RAG system?",
    "What are the components of the system architecture?",
    "Which workload does the system aim to reduce?",
    "What is the capital of Australia?",  # off-topic: should say it doesn't know
]
TASK2_TURNS = [
    "What is the purpose of the RAG system described in this paper?",
    "What problem does it try to solve?",
    "How does the system do that?",            # follow-up: needs memory
    "What was the first thing I asked you?",   # tests memory directly
    "Summarise our conversation so far in two sentences.",
]
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------- vector store
class DocStore:
    """PDF -> chunks -> embeddings -> ChromaDB (cosine) with top-k retrieval."""

    def __init__(self, pdf_path: str):
        self.emb = OllamaEmbeddings(model=EMBED_MODEL)
        client = chromadb.PersistentClient(path="./chroma_db")
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        try:
            self.col = client.create_collection(
                COLLECTION, configuration={"hnsw": {"space": "cosine"}}
            )
        except Exception:  # older Chroma versions
            self.col = client.create_collection(
                COLLECTION, metadata={"hnsw:space": "cosine"}
            )
        self._ingest(pdf_path)

    def _ingest(self, pdf_path: str):
        reader = PdfReader(pdf_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts, metas = [], []
        for page_no, page in enumerate(reader.pages, start=1):
            for chunk in splitter.split_text(page.extract_text() or ""):
                texts.append(chunk)
                metas.append({"page": page_no})
        if not texts:
            raise SystemExit("No text extracted. Is the PDF scanned (image only)?")
        self.col.add(
            ids=[f"chunk{i}" for i in range(len(texts))],
            documents=texts,
            embeddings=self.emb.embed_documents(texts),
            metadatas=metas,
        )
        print(f"Ingested {len(reader.pages)} pages into {len(texts)} chunks.")

    def context_for(self, question: str) -> str:
        res = self.col.query(
            query_embeddings=[self.emb.embed_query(question)], n_results=TOP_K
        )
        parts = []
        for text, meta in zip(res["documents"][0], res["metadatas"][0]):
            parts.append(f"(page {meta['page']}) {text}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------- Task 1: chain
def build_rag_chain(store: DocStore):
    prompt = PromptTemplate.from_template(
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say "
        "\"I don't know based on the document.\"\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    llm = OllamaLLM(model=CHAT_MODEL, temperature=0)
    return (
        {
            "context": RunnableLambda(store.context_for),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def run_task1(store: DocStore):
    chain = build_rag_chain(store)
    print("\n=== Task 1: RAG chain with 5 questions ===")
    for i, q in enumerate(TASK1_QUESTIONS, 1):
        print(f"\n[{i}] Q: {q}\n    A: {chain.invoke(q).strip()}")


# ---------------------------------------------------------------- Task 2: memory
class DocChatbot:
    """Document chatbot with a conversation buffer (full message history).

    ConversationBufferMemory is deprecated, so the buffer is a plain list of
    messages, which is exactly what that class stored.
    """

    def __init__(self, store: DocStore):
        self.store = store
        self.history: list = []
        llm = ChatOllama(model=CHAT_MODEL, temperature=0)

        condense_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Rewrite the user's latest message as ONE standalone search "
                 "question, resolving words like 'it', 'that' or 'the system' "
                 "using the chat history. Do NOT answer it. Output exactly one "
                 "line ending with a question mark.\n"
                 "Example: history says the user asked about the Eiffel Tower; "
                 "latest message 'How tall is it?' -> 'How tall is the Eiffel Tower?'"),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are a document assistant. Use the context below for facts "
                 "about the document. If the document does not contain the answer, "
                 "say \"I don't know based on the document.\" You may use the chat "
                 "history to answer questions about the conversation itself.\n\n"
                 "Context:\n{context}"),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )
        self.condense = condense_prompt | llm | StrOutputParser()
        self.answer = answer_prompt | llm | StrOutputParser()

    def _valid_rewrite(self, raw: str, question: str) -> str:
        """Small models sometimes answer instead of rewriting; guard against it."""
        lines = raw.strip().splitlines()
        line = lines[0].strip() if lines else ""
        if line.endswith("?") and len(line.split()) <= 30:
            return line
        # Fallback: previous user question + current question (no LLM needed).
        last_user = next(
            (m.content for m in reversed(self.history) if isinstance(m, HumanMessage)),
            "",
        )
        return f"{last_user} {question}".strip()

    def ask(self, question: str) -> tuple[str, str]:
        # Rewrite follow-ups so retrieval searches for the real topic.
        standalone = question
        if self.history:
            raw = self.condense.invoke(
                {"history": self.history, "question": question}
            )
            standalone = self._valid_rewrite(raw, question)
        context = self.store.context_for(standalone)
        reply = self.answer.invoke(
            {"context": context, "history": self.history, "question": question}
        ).strip()
        self.history.extend([HumanMessage(question), AIMessage(reply)])
        return reply, standalone


def run_task2(store: DocStore):
    bot = DocChatbot(store)
    print("\n\n=== Task 2: memory across 5 turns ===")
    for i, q in enumerate(TASK2_TURNS, 1):
        reply, standalone = bot.ask(q)
        print(f"\nTurn {i}\n  You       : {q}")
        if standalone != q:
            print(f"  Searched  : {standalone}")
        print(f"  Bot       : {reply}")

    print(f"\nStored messages: {len(bot.history)} (expected 10)")
    assert len(bot.history) == 10, "History not maintained!"
    for m in bot.history:
        print(f"  {m.type:>5}: {m.content[:70]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", default="sample.pdf")
    args = parser.parse_args()

    doc_store = DocStore(args.pdf)
    run_task1(doc_store)
    run_task2(doc_store)