"""W6D3 Tasks 1 & 2: PromptTemplate -> Ollama LLM -> OutputParser, then add memory.

Prereqs:
    pip install langchain langchain-core langchain-ollama
    ollama pull llama3.2
"""
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama, OllamaLLM

MODEL = "llama3.2"


# ---------------------------------------------------------------- Task 1: chain
def run_chain():
    prompt = PromptTemplate.from_template(
        "Explain the concept of '{concept}' in exactly two sentences "
        "for a beginner."
    )
    llm = OllamaLLM(model=MODEL, temperature=0)
    parser = StrOutputParser()

    chain = prompt | llm | parser  # LCEL: prompt -> model -> parser

    inputs = ["overfitting", "embeddings", "recursion", "REST API", "vector database"]
    print("=== Task 1: chain with 5 inputs ===")
    for i, concept in enumerate(inputs, 1):
        out = chain.invoke({"concept": concept})
        print(f"\n[{i}] {concept}\n{out.strip()}")


# ---------------------------------------------------------------- Task 2: memory
_store: dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


def run_memory_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise, friendly assistant."),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    llm = ChatOllama(model=MODEL, temperature=0)
    chain = prompt | llm | StrOutputParser()

    # RunnableWithMessageHistory is the modern replacement for the deprecated
    # ConversationBufferMemory: it keeps the FULL buffer of messages per session.
    chat = RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    cfg = {"configurable": {"session_id": "demo"}}

    turns = [
        "Hi, my name is Siriyala and I'm learning LangChain.",
        "My favourite language is Python.",
        "I'm building a RAG project this week.",
        "What is my name?",
        "Summarise everything you know about me in one sentence.",
    ]

    print("\n\n=== Task 2: memory across 5 turns ===")
    for i, msg in enumerate(turns, 1):
        reply = chat.invoke({"input": msg}, config=cfg)
        print(f"\nTurn {i}\n  You: {msg}\n  AI : {reply.strip()}")

    # Verification: 5 human + 5 AI messages should be stored.
    history = get_history("demo").messages
    print(f"\nStored messages: {len(history)} (expected 10)")
    assert len(history) == 10, "History not maintained!"
    for m in history:
        print(f"  {m.type:>5}: {m.content[:70]}")


if __name__ == "__main__":
    run_chain()
    run_memory_chain()
