import json
from pathlib import Path
from typing import TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph

from .config import OLLAMA_BASE_URL, CHAT_MODEL, EMBED_MODEL


class RAGState(TypedDict, total=False):
    question: str
    contexts: list[str]
    answer: str
    review: str


class TrackedRAG:
    def __init__(self, data_path: str = "data/knowledge_base.json"):
        self.data_path = Path(data_path)
        self.embeddings = OllamaEmbeddings(
            model=EMBED_MODEL, base_url=OLLAMA_BASE_URL
        )
        self.llm = ChatOllama(
            model=CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0
        )
        self.vectorstore = self._build_vectorstore()
        self.graph = self._build_graph()

    def _build_vectorstore(self):
        records = json.loads(self.data_path.read_text(encoding="utf-8"))
        docs = [
            Document(
                page_content=item["content"],
                metadata={"id": item["id"], "topic": item["topic"]},
            )
            for item in records
        ]
        return Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name="w11d5_knowledge",
            persist_directory="chroma_db",
        )

    def _retrieve(self, state: RAGState) -> RAGState:
        docs = self.vectorstore.similarity_search(state["question"], k=3)
        return {"contexts": [doc.page_content for doc in docs]}

    def _generate(self, state: RAGState) -> RAGState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer only from the supplied context. If it does not contain the answer, say you do not have enough information. Do not invent policy details."),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        chain = prompt | self.llm
        result = chain.invoke({
            "context": "\n\n".join(state.get("contexts", [])),
            "question": state["question"],
        })
        return {"answer": result.content}

    def _build_graph(self):
        builder = StateGraph(RAGState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    def ask(self, question: str) -> dict:
        result = self.graph.invoke({"question": question})
        return {
            "question": question,
            "answer": result["answer"],
            "contexts": result.get("contexts", []),
        }
