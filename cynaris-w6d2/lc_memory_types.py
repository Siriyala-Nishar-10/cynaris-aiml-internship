"""
W6D2: Conversation memory strategies with LangGraph (buffer, window, summary)
------------------------------------------------------------------------------
Memory keeps the conversation so the model can use it on later turns. The
problem is size: every turn adds to the history, so a full "buffer" grows
until it slows the model down and finally overflows its context window.
Three common strategies trade recall against prompt size:

  buffer    send the WHOLE history every turn       (best recall, grows forever)
  window    send only the last few messages         (small, forgets old facts)
  summary   send a running summary + the last few   (small, keeps the gist)

In LangChain 1.x the old memory classes (ConversationBufferMemory,
RunnableWithMessageHistory) are deprecated in favour of LangGraph persistence:
the conversation lives in the graph state, and a CHECKPOINTER saves that state
per thread_id after every step. This script uses that current approach.

What it does:

  1. Runs the same 10-turn conversation through each strategy. Turns 1 to 4
     state facts (name, project, favourite language, setup), turns 5 to 8 are
     filler questions, and turns 9 and 10 ask about the early facts.
  2. Measures how many characters of context each strategy sends per turn
     (about 4 characters = 1 token) and whether turns 9 and 10 are answered
     correctly.
  3. Persistence: saves a conversation to a SQLite file, opens a brand-new
     connection (as if the program had restarted) and continues the same
     thread, to show the history survives.

Output:
  memory_types_results.json   per-strategy sizes, recall checks and persistence result

Run:  python lc_memory_types.py
      python lc_memory_types.py --model qwen2.5:3b
Needs: pip install langgraph langchain-ollama langgraph-checkpoint-sqlite
       ollama pull llama3.2:3b   (Ollama must be running)
       Takes several minutes on CPU: 3 strategies x 10 turns of model calls.
"""

import argparse
import json
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

MODEL = "llama3.2:3b"
OUTPUT_FILE = "memory_types_results.json"
DB_FILE = "chat_memory.sqlite"
WINDOW = 4          # messages kept verbatim by the window and summary strategies

SYSTEM = "You are a friendly assistant. Answer in at most two sentences."

TURNS = [
    "Hi, my name is Siriyala.",
    "I'm building a support-ticket triage bot.",
    "My favourite programming language is Python.",
    "I run everything locally with Ollama.",
    "Explain what a vector database is in one sentence.",
    "Explain cosine similarity in one sentence.",
    "What is semantic caching, in one sentence?",
    "Give me a one-sentence definition of RAG.",
    "What is my name?",                                             # probe 1
    "What am I building, and which language do I like?",            # probe 2
]
PROBES = {   # turn number -> words the answer must contain
    9: ["siriyala"],
    10: ["triage", "python"],
}


class ChatState(MessagesState):
    """Messages (with the add-messages reducer) plus the summary bookkeeping."""
    summary: str
    summarized_upto: int     # messages before this index are covered by the summary
    context_chars: int       # size of the context sent to the model this turn


def plan_context(strategy: str, n_messages: int, upto: int, window: int):
    """Decide what the model sees. Returns (keep_from, range_to_summarise or None).

    Messages from index keep_from onward are sent verbatim. For the summary
    strategy, messages in [upto, keep_from) must be folded into the summary.
    """
    if strategy == "buffer":
        return 0, None
    keep_from = max(0, n_messages - window)
    if strategy == "window":
        return keep_from, None
    return keep_from, ((upto, keep_from) if keep_from > upto else None)


def make_graph(strategy: str, checkpointer, model: str):
    """One-node chat graph; the strategy only changes what is sent to the model."""
    llm = ChatOllama(model=model, temperature=0)

    def summarise(old_summary: str, messages: list) -> str:
        transcript = "\n".join(f"{m.type}: {m.content}" for m in messages)
        prompt = (
            "Update the running summary of a conversation. Keep every fact the user "
            "stated about themselves (name, project, preferences, setup). "
            "Reply with the summary only, in at most 3 sentences.\n\n"
            f"Existing summary: {old_summary or '(none)'}\n\nNew messages:\n{transcript}"
        )
        return llm.invoke([HumanMessage(prompt)]).content.strip()

    def chat(state: ChatState) -> dict:
        messages = state["messages"]
        summary = state.get("summary", "")
        upto = state.get("summarized_upto", 0)

        keep_from, to_fold = plan_context(strategy, len(messages), upto, WINDOW)
        if to_fold:
            summary = summarise(summary, messages[to_fold[0]:to_fold[1]])
            upto = to_fold[1]

        context = [SystemMessage(SYSTEM)]
        if strategy == "summary" and summary:
            context.append(SystemMessage(f"Summary of the earlier conversation: {summary}"))
        context += messages[keep_from:]

        reply = llm.invoke(context)
        return {
            "messages": [reply],
            "summary": summary,
            "summarized_upto": upto,
            "context_chars": sum(len(str(m.content)) for m in context),
        }

    builder = StateGraph(ChatState)
    builder.add_node("chat", chat)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)


def run_strategy(strategy: str, model: str) -> dict:
    """Run the 10-turn conversation with one strategy on its own thread."""
    graph = make_graph(strategy, InMemorySaver(), model)
    config = {"configurable": {"thread_id": f"{strategy}-demo"}}

    sizes, answers = [], []
    start = time.perf_counter()
    state = {}
    for n, question in enumerate(TURNS, start=1):
        state = graph.invoke({"messages": [HumanMessage(question)]}, config)
        answer = state["messages"][-1].content
        sizes.append(state["context_chars"])
        answers.append(answer)
        print(f"  turn {n:>2} | context {state['context_chars']:>5} chars | {answer[:80]!r}")
    seconds = round(time.perf_counter() - start, 1)

    recall = {turn: all(w in answers[turn - 1].lower() for w in words) for turn, words in PROBES.items()}
    return {"strategy": strategy, "context_chars_per_turn": sizes, "recall": recall,
            "answers": answers, "summary": state.get("summary", ""), "seconds": seconds}


def persistence_demo(model: str) -> dict:
    """Save a conversation to SQLite, reopen it with a new connection, and continue it."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        print("  skipped: pip install langgraph-checkpoint-sqlite")
        return {"skipped": True}

    thread = f"persist-{int(time.time())}"
    config = {"configurable": {"thread_id": thread}}

    with SqliteSaver.from_conn_string(DB_FILE) as saver:           # session 1
        graph = make_graph("buffer", saver, model)
        graph.invoke({"messages": [HumanMessage("Hi, my name is Siriyala and I like Python.")]}, config)
        saved = len(graph.get_state(config).values["messages"])
    print(f"  session 1 saved {saved} messages to {DB_FILE} (thread {thread})")

    with SqliteSaver.from_conn_string(DB_FILE) as saver:           # session 2: fresh connection
        graph = make_graph("buffer", saver, model)
        loaded = len(graph.get_state(config).values["messages"])
        out = graph.invoke({"messages": [HumanMessage("What is my name and which language do I like?")]}, config)
        answer = out["messages"][-1].content
    ok = loaded == saved and "siriyala" in answer.lower() and "python" in answer.lower()
    print(f"  session 2 loaded {loaded} messages, then answered: {answer!r}")
    print(f"  history survived the restart: {'PASS' if ok else 'CHECK'}")
    return {"thread": thread, "saved": saved, "loaded": loaded, "answer": answer, "passed": ok}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()

    print(f"Model: {args.model} | window = last {WINDOW} messages\n")
    results = []
    for strategy in ("buffer", "window", "summary"):
        print(f"=== {strategy} ===")
        results.append(run_strategy(strategy, args.model))
        print()

    print("=" * 78)
    print(f"{'strategy':<10}{'context at turn 10':>20}{'turn 9 (name)':>16}{'turn 10 (project, language)':>30}{'seconds':>9}")
    for r in results:
        t9 = "PASS" if r["recall"][9] else "FORGOT"
        t10 = "PASS" if r["recall"][10] else "FORGOT"
        print(f"{r['strategy']:<10}{r['context_chars_per_turn'][-1]:>16} chars{t9:>16}{t10:>30}{r['seconds']:>9}")

    summary_run = next(r for r in results if r["strategy"] == "summary")
    print(f"\nSummary kept by the summary strategy:\n  {summary_run['summary']!r}")

    print("\n=== persistence (SQLite checkpointer) ===")
    persistence = persistence_demo(args.model)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "window": WINDOW, "strategies": results,
                   "persistence": persistence}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {OUTPUT_FILE}")
    print("VERIFY: buffer recalls everything but sends the most text; window forgets the early facts; "
          "summary recalls them from a short summary; the SQLite thread resumes after reopening")


if __name__ == "__main__":
    main()
