"""
W6D1 (Task 2): Conversation memory across 5 turns
---------------------------------------------------
An LLM call is stateless: the model only knows what is in the prompt. Memory
solves this by storing the conversation and putting it back into every new
prompt, so the model can answer "what is my name?" a few turns later.

The lesson names ConversationBufferMemory (store the whole conversation,
unchanged). In LangChain 1.x that class lives in the legacy `langchain-classic`
package and is deprecated in favour of RunnableWithMessageHistory, which does
the same job: an InMemoryChatMessageHistory keeps every message and a
MessagesPlaceholder puts them back into the prompt. This script uses the
current approach, then shows the legacy class too if it is installed.

What this script does:

  1. Runs a 5-turn conversation with history stored per session id.
  2. Checks that turn 3 ("what is my name?") is answered correctly.
  3. Control: asks the same question in a NEW session with no history, to show
     that without memory the model cannot know.
  4. Prints the stored history (5 turns = 10 messages).
  5. Optional: the legacy ConversationBufferMemory API on the same turns.

Output:
  memory_results.json   turns, answers, history length and checks

Run:  python lc_memory.py
Needs: pip install langchain langchain-ollama
       optional: pip install langchain-classic   (for the legacy class demo)
       ollama pull llama3.2:3b
"""

import argparse
import json
import warnings

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

MODEL = "llama3.2:3b"
OUTPUT_FILE = "memory_results.json"

TURNS = [
    "Hi, my name is Siriyala and I'm learning LangChain.",
    "I'm running a local model through Ollama on my laptop.",
    "What is my name?",
    "What am I learning, and how am I running the model?",
    "Summarise our conversation so far in one sentence.",
]

store: dict[str, InMemoryChatMessageHistory] = {}   # one history per session id


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the history for a session, creating it on first use."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def build_chat(model: str) -> RunnableWithMessageHistory:
    """prompt (with a history slot) -> chat model -> parser, wrapped with memory."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly assistant. Answer briefly, in at most two sentences."),
        MessagesPlaceholder("history"),     # past messages are inserted here
        ("human", "{input}"),
    ])
    chain = prompt | ChatOllama(model=model, temperature=0) | StrOutputParser()
    return RunnableWithMessageHistory(
        chain, get_session_history,
        input_messages_key="input", history_messages_key="history",
    )


def legacy_buffer_memory_demo() -> str:
    """The lesson's ConversationBufferMemory, if the legacy package is installed."""
    warnings.filterwarnings("ignore")
    try:
        from langchain_classic.memory import ConversationBufferMemory
    except ImportError:
        try:
            from langchain.memory import ConversationBufferMemory
        except ImportError:
            return "ConversationBufferMemory not available (pip install langchain-classic to try it)"

    memory = ConversationBufferMemory(return_messages=True)
    memory.save_context({"input": TURNS[0]}, {"output": "Nice to meet you, Siriyala!"})
    memory.save_context({"input": TURNS[1]}, {"output": "Great, local models are private and free to run."})
    history = memory.load_memory_variables({})["history"]
    return f"ConversationBufferMemory holds {len(history)} messages after 2 turns"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()

    chat = build_chat(args.model)
    config = {"configurable": {"session_id": "demo"}}

    print(f"Model: {args.model} | session: demo\n")
    answers = []
    for n, question in enumerate(TURNS, start=1):
        answer = chat.invoke({"input": question}, config=config)
        answers.append(answer)
        print(f"Turn {n}  You: {question}\n        Bot: {answer}\n")

    history = get_session_history("demo").messages
    name_ok = "siriyala" in answers[2].lower()
    topic_ok = "langchain" in answers[3].lower() and "ollama" in answers[3].lower()

    print("=" * 70)
    print(f"History stored: {len(history)} messages (expected 10 = 5 turns x 2)")
    print(f"Turn 3 remembers the name:      {'PASS' if name_ok else 'CHECK'}")
    print(f"Turn 4 remembers topic + tool:  {'PASS' if topic_ok else 'CHECK'}")

    print("\nCONTROL: same question in a NEW session, no history")
    control = chat.invoke({"input": TURNS[2]}, config={"configurable": {"session_id": "fresh"}})
    control_forgot = "siriyala" not in control.lower()
    print(f"  You: {TURNS[2]}\n  Bot: {control}")
    print(f"  Does not know the name without memory: {'YES' if control_forgot else 'NO (check)'}")

    print("\nStored history:")
    for m in history:
        print(f"  {m.type:>5}: {m.content[:90]}")

    legacy = legacy_buffer_memory_demo()
    print(f"\nLegacy class: {legacy}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "turns": [{"question": q, "answer": a} for q, a in zip(TURNS, answers)],
            "history_messages": len(history),
            "turn3_remembers_name": name_ok,
            "turn4_remembers_topic": topic_ok,
            "control_answer": control, "control_forgot": control_forgot,
            "legacy_demo": legacy,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {OUTPUT_FILE}")
    print("VERIFY: turn 3 answers 'Siriyala', and the control session does not")


if __name__ == "__main__":
    main()
