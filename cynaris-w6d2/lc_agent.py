"""
W6D1 (Task 3): LangChain agent with 2 tools (web search stub + calculator)
---------------------------------------------------------------------------
An agent lets the model decide WHICH tool to call and WHEN, instead of
following a fixed chain. It follows the ReAct pattern (Reason + Act):

  Thought      the model reasons about what it still needs
  Action       it calls a tool with arguments (web_search or calculator)
  Observation  the tool's result is fed back to the model
  ...repeat... until the model has enough to give a Final Answer

In LangChain 1.x, `create_agent` runs this loop (it is built on LangGraph) and
the model requests tools through tool calling. The trace printed below maps
straight onto ReAct: an AIMessage with tool_calls is the Action, a ToolMessage
is the Observation, and the last AIMessage is the Final Answer.

Tools:
  web_search   a STUB: looks up a small built-in dictionary, no real internet
  calculator   evaluates arithmetic safely with the ast module (no eval)

The 3 tasks cover: search only, calculator only, and both tools in one task.

Output:
  agent_results_<model>.json   tasks, tool trace, final answers and checks
                               (one file per model, e.g. agent_results_qwen2.5_3b.json)

Run:  python lc_agent.py
      python lc_agent.py --model qwen2.5:3b
Needs: pip install langchain langchain-ollama
       ollama pull llama3.2:3b
Note:  small models can pick the wrong tool or mangle arguments. If a task
       fails, try --model qwen2.5:3b or a larger model such as llama3.1:8b.
"""

import argparse
import ast
import json
import operator
import re
import time

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are a helpful assistant with two tools.\n"
    "Rules:\n"
    "1. Use web_search to look up any fact you do not already have, such as a height, "
    "a capital or a population. Never guess facts.\n"
    "2. Use calculator for ANY arithmetic, with numbers only, for example '8849 * 3.28084'. "
    "Never put words in the expression.\n"
    "3. For a question with two parts, call web_search first, then call calculator "
    "with the number you found.\n"
    "4. Call a tool only when you need it. Do not call a tool whose result you already have.\n"
    "5. Finish with a short answer in a full sentence that states the facts you found."
)

# ---- Tool 1: web search STUB ------------------------------------------------
STUB_RESULTS = {
    "capital of france": "Paris is the capital and largest city of France.",
    "height of mount everest": "Mount Everest is 8,849 metres tall (2020 survey).",
    "population of india": "India has a population of about 1.43 billion people.",
    "tallest building in the world": "The Burj Khalifa in Dubai is the tallest building, at 828 metres.",
}


@tool
def web_search(query: str) -> str:
    """Search the web for a fact. Input is a short search query, for example 'capital of France'."""
    words = set(re.findall(r"[a-z]+", query.lower()))
    best_key, best_overlap = None, 0
    for key in STUB_RESULTS:
        overlap = len(words & set(key.split()))
        if overlap > best_overlap:
            best_key, best_overlap = key, overlap
    if best_key and best_overlap >= 2:
        return STUB_RESULTS[best_key]
    return "No results found."


# ---- Tool 2: calculator (safe arithmetic) -----------------------------------
OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(node):
    """Evaluate a parsed arithmetic expression; anything else is rejected."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, right = safe_eval(node.left), safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](safe_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, for example '1234 * 5678 + 91'.
    Supports + - * / // % ** and parentheses. Numbers only, no words."""
    try:
        result = safe_eval(ast.parse(expression.strip(), mode="eval").body)
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, SyntaxError):
        return ("Error: invalid expression. Use numbers and operators only, "
                "for example '8849 * 3.28084'. Look up any missing number with web_search first.")
    return str(round(result, 6) if isinstance(result, float) else result)


# ---- The 3 tasks: (task, text the final answer must contain, tools it must use) ----
# A task only PASSES if the answer is right AND the right tools were called, so a
# correct number that came from the model's memory instead of the tool does not count.
TASKS = [
    ("What is the capital of France?", "paris", {"web_search"}),
    ("What is 1234 * 5678 + 91?", "7006743", {"calculator"}),
    ("How tall is Mount Everest in metres, and what is that in feet "
     "(multiply metres by 3.28084)?", "29032", {"web_search", "calculator"}),
]


def trace(messages: list) -> list[dict]:
    """Turn the message list into ReAct-style steps: Action, Observation, Final answer."""
    steps = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for call in m.tool_calls:
                steps.append({"type": "Action", "text": f"{call['name']}({call['args']})"})
        elif isinstance(m, ToolMessage):
            steps.append({"type": "Observation", "text": str(m.content)})
        elif isinstance(m, AIMessage):
            steps.append({"type": "Final answer", "text": str(m.content)})
    return steps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()

    agent = create_agent(
        model=ChatOllama(model=args.model, temperature=0),
        tools=[web_search, calculator],
        system_prompt=SYSTEM_PROMPT,
    )

    print(f"Model: {args.model} | tools: web_search (stub), calculator\n")
    results = []
    for n, (task, expected, tools_needed) in enumerate(TASKS, start=1):
        print("=" * 70 + f"\nTask {n}: {task}")
        start = time.perf_counter()
        try:
            out = agent.invoke({"messages": [("user", task)]}, config={"recursion_limit": 12})
            steps = trace(out["messages"])
            error = None
        except Exception as exc:   # e.g. model without tool support, loop limit
            steps, error = [], str(exc)
        seconds = round(time.perf_counter() - start, 1)

        for s in steps:
            print(f"  {s['type']}: {s['text']}")
        final = next((s["text"] for s in reversed(steps) if s["type"] == "Final answer"), "")
        used = [s["text"].split("(")[0] for s in steps if s["type"] == "Action"]
        text_ok = expected in final.lower().replace(",", "")
        tools_ok = tools_needed <= set(used)
        passed = text_ok and tools_ok
        if error:
            print(f"  ERROR: {error}")
        print(f"  Tools used: {used or 'none'} | {seconds}s")
        print(f"  Answer contains {expected!r}: {'yes' if text_ok else 'NO'} | "
              f"needed tools {sorted(tools_needed)} used: {'yes' if tools_ok else 'NO'} | "
              f"{'PASS' if passed else 'CHECK'}\n")
        results.append({"task": task, "steps": steps, "final": final, "tools_used": used,
                        "expected": expected, "tools_needed": sorted(tools_needed),
                        "text_ok": text_ok, "tools_ok": tools_ok, "passed": passed,
                        "seconds": seconds, "error": error})

    output_file = f"agent_results_{args.model.replace(':', '_')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved {output_file}")
    print(f"Passed {sum(r['passed'] for r in results)}/{len(results)} tasks")
    print("VERIFY: task 1 uses web_search, task 2 uses calculator, task 3 uses web_search then calculator")


if __name__ == "__main__":
    main()