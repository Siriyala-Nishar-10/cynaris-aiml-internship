"""Two simple tools for the W9D4 agent: a safe web-search stub and calculator."""
import ast
import operator as op
from langchain_core.tools import tool

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_calculate(expression: str):
    """Evaluate arithmetic only; never use eval()."""
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](visit(node.operand))
        raise ValueError("Only basic arithmetic expressions are allowed.")
    tree = ast.parse(expression, mode="eval")
    return visit(tree)


@tool
def web_search_stub(query: str) -> str:
    """Return sample search information for a topic. This is a demo stub, not live web search."""
    sample = {
        "langchain": "LangChain is a framework for composing LLM applications with prompts, models, parsers, tools, and memory integrations.",
        "crewai": "CrewAI is a framework for coordinating role-based AI agents that collaborate on tasks.",
        "rag": "Retrieval-augmented generation retrieves relevant external information and supplies it to a language model as context.",
    }
    lowered = query.lower()
    for key, value in sample.items():
        if key in lowered:
            return f"Demo search result for '{query}': {value}"
    return (
        f"Demo search result for '{query}': No live search was performed. "
        "Replace this stub with an approved search API if required."
    )


@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression, for example '25 * 4 + 10'."""
    try:
        return str(_safe_calculate(expression))
    except Exception as exc:
        return f"Calculator error: {exc}"
