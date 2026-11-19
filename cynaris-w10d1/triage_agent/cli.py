"""Command line: demo, chat, diagram.

Console output is plain ASCII on purpose: it prints correctly in any Windows
terminal encoding.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

from langgraph.types import Command

from .classify import LLMClassifier, RuleClassifier
from .graph import build_graph
from .llm_client import ollama_chat_fn

DEMO_INPUTS = [
    "I was charged twice on my last invoice",
    "The app crashes every time I open it",
    "I want a refund, this product is broken",
    "Hello, what are your opening hours?",
    "asdf qwerty",
]


def build_classifier(use_llm: bool):
    rules = RuleClassifier()
    if not use_llm:
        return rules
    chat = ollama_chat_fn(os.getenv("TRIAGE_MODEL", "llama3.2:3b"),
                          os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    return LLMClassifier(chat, fallback=rules)


def process_message(graph, message: str, thread_id: str, ask_human) -> dict:
    """Run one ticket to completion, pausing for a human whenever the graph asks.

    Each ticket gets its own thread_id: state is saved per thread, so reusing one
    id would carry the previous ticket's trace into the next.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"message": message, "trace": []}, config)
    while "__interrupt__" in result:           # graph is paused
        decision = ask_human(result["__interrupt__"][0].value)
        result = graph.invoke(Command(resume=decision), config)  # continue from the pause
    return result


def ask_human_console(payload: dict) -> dict:
    print(f"\n  [PAUSED] human review needed: {payload['reason']}")
    print(f"    message : {payload['message']}")
    print(f"    category: {payload['label']} (confidence {payload['confidence']})")
    approved = input("    Approve automated handling? [y/N]: ").strip().lower() == "y"
    note = input("    Note / answer for the customer (optional): ").strip()
    return {"approved": approved, "note": note}


def cmd_demo(graph) -> int:
    print(f"{'#':<2} {'label':<10} {'conf':<5} {'route':<6} {'path':<52} message")
    for number, message in enumerate(DEMO_INPUTS, start=1):
        config = {"configurable": {"thread_id": f"demo-{number}"}}
        result = graph.invoke({"message": message, "trace": []}, config)
        state = graph.get_state(config)
        paused = bool(state.next)
        values = state.values
        conf = "-" if values.get("confidence") is None else f"{values['confidence']:.2f}"
        path = " > ".join(values["trace"]) + (f" > [PAUSED at {state.next[0]}]" if paused else "")
        print(f"{number:<2} {values['label']:<10} {conf:<5} {values['route']:<6} {path:<52} {message}")
        if not paused:
            print(f"   reply: {result['response']}")
    print("\n[PAUSED] = waiting for a human. Try `chat` to resume one interactively.")
    return 0


def cmd_chat(graph) -> int:
    print("Type a customer message (blank line to quit).")
    for number in itertools.count(1):
        message = input("\ncustomer> ").strip()
        if not message:
            return 0
        result = process_message(graph, message, f"ticket-{number}", ask_human_console)
        print(f"\n  path : {' > '.join(result['trace'])}")
        print(f"  reply: {result['response']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="triage_agent")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "chat"):
        sub.add_parser(name).add_argument("--llm", action="store_true",
                                          help="classify with a local Ollama model")
    diagram = sub.add_parser("diagram", help="write the graph as a Mermaid diagram")
    diagram.add_argument("--out", default="graph.mmd")
    args = parser.parse_args(argv)

    if args.command == "diagram":
        graph = build_graph(RuleClassifier())
        Path(args.out).write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
        print(f"Wrote {args.out}")
        return 0

    graph = build_graph(build_classifier(args.llm))
    try:
        return cmd_demo(graph) if args.command == "demo" else cmd_chat(graph)
    except RuntimeError as exc:   # e.g. Ollama not running
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
