"""W6D3 Task 3: agent with 2 tools (web search stub + calculator), run 3 tasks.

Prereqs:
    pip install langchain langchain-ollama langchain-core
    ollama pull llama3.2      # llama3.2 supports tool calling
"""
import ast
import operator as op

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
    "Search first, then calculate, then give a short final answer."
)

llm = ChatOllama(model="llama3.2", temperature=0)
agent = create_agent(llm, [web_search, calculator], system_prompt=SYSTEM_PROMPT)

TASKS = [
    "What is 23 * 47 + 130?",                                   # calculator only
    "What is the height of Mount Everest?",                     # search only
    "Find the height of Mount Everest and tell me how many "    # both tools
    "times taller it is than a 2-metre person.",
]


def run(task: str):
    print(f"\n{'=' * 60}\nTASK: {task}")
    result = agent.invoke({"messages": [("user", task)]})
    tools_used = []
    for m in result["messages"]:
        if m.type == "ai" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                tools_used.append(tc["name"])
                print(f"  -> ACTION: {tc['name']}({tc['args']})")
        elif m.type == "tool":
            print(f"  <- OBSERVATION: {m.content}")
    print(f"TOOLS USED: {tools_used or 'none'}")
    print(f"FINAL ANSWER: {result['messages'][-1].content}")


if __name__ == "__main__":
    for t in TASKS:
        run(t)