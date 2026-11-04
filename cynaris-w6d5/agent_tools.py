"""W6D3 Task 3: agent with 2 tools (web search stub + calculator), run 3 tasks.

Prereqs:
    pip install langchain langchain-ollama langchain-core
    ollama pull llama3.2      # llama3.2 supports tool calling
"""
import ast
import operator as op
import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------- tools
_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg,
}


def _eval(node):
    """Safely evaluate arithmetic only (no eval() on raw strings)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression such as '12 * (3 + 4) / 2'. Use for ANY arithmetic."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


@tool
def web_search(query: str) -> str:
    """Search the web for facts. Use for questions about people, places or data."""
    # Stub: canned results so the agent is testable offline.
    fake_db = {
        "population of india": "India's population is about 1.43 billion (2023 estimate).",
        "capital of france": "The capital of France is Paris.",
        "height of mount everest": "Mount Everest is 8,849 metres tall.",
    }
    q = query.lower()
    for key, val in fake_db.items():
        if key in q:
            return val
    return f"[stub] No stored result for '{query}'."


# ---------------------------------------------------------------- agent
SYSTEM_PROMPT = (
    "You are a helpful assistant with two tools. "
    "Use web_search for facts. "
    "Use the calculator tool for ALL arithmetic, even simple division. "
    "Never do math in your head. "
    "Call the calculator ONLY when the question asks for a calculation. "
    "If the question only asks for a fact, call web_search once and answer "
    "with that fact; do not invent extra calculations or assumptions. "
    "If it needs a fact AND arithmetic, first call web_search, then call "
    "calculator using only numbers from the question and the search result. "
    "'How many times taller' means DIVIDE (a / b)."
)

AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen2.5:3b")  # llama3.2 failed 2 of 3 tasks
print(f"Agent model: {AGENT_MODEL}")
llm = ChatOllama(model=AGENT_MODEL, temperature=0)
agent = create_agent(llm, [web_search, calculator], system_prompt=SYSTEM_PROMPT)

# (task, tools the agent must use, text the final answer must contain)
TASKS = [
    ("What is 23 * 47 + 130?", {"calculator"}, "1211"),
    ("What is the height of Mount Everest?", {"web_search"}, "8849"),
    ("Find the height of Mount Everest and tell me how many "
     "times taller it is than a 2-metre person.",
     {"web_search", "calculator"}, "4424"),
]


def run(task: str, expected: set, must_contain: str):
    print(f"\n{'=' * 60}\nTASK: {task}")
    result = agent.invoke({"messages": [("user", task)]})
    tools_used, calc_errors = [], 0
    for m in result["messages"]:
        if m.type == "ai" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                tools_used.append(tc["name"])
                print(f"  -> ACTION: {tc['name']}({tc['args']})")
        elif m.type == "tool":
            print(f"  <- OBSERVATION: {m.content}")
            if getattr(m, "name", "") == "calculator" and str(m.content).startswith("Error"):
                calc_errors += 1
    print(f"TOOLS USED: {tools_used or 'none'}")
    final = str(result["messages"][-1].content)
    problems = []
    if not expected.issubset(set(tools_used)):
        problems.append(f"missing tools {sorted(expected - set(tools_used))}")
    if calc_errors:
        problems.append(f"calculator returned {calc_errors} error(s)")
    if must_contain not in final.replace(",", "").replace(" ", ""):
        problems.append(f"answer missing expected value {must_contain}")
    print(f"CHECK: {'PASS' if not problems else 'FAIL - ' + '; '.join(problems)}")
    print(f"FINAL ANSWER: {final}")


if __name__ == "__main__":
    for t, expected, must in TASKS:
        run(t, expected, must)