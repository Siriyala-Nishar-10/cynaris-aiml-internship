
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

Category = Literal["technical", "account", "general"]

class AgentState(TypedDict, total=False):
    user_input: str
    category: Category
    human_feedback: str
    final_response: str

def classify_node(state: AgentState) -> dict:
    message = state["user_input"].lower()
    if any(word in message for word in ("python", "code", "error", "bug", "langgraph", "install", "pytest")):
        return {"category": "technical"}
    if any(word in message for word in ("login", "password", "account", "billing", "payment", "worklog")):
        return {"category": "account"}
    return {"category": "general"}

def route_by_category(state: AgentState) -> str:
    return {
        "technical": "respond_technical",
        "account": "respond_account",
        "general": "respond_general",
    }.get(state.get("category", "general"), "respond_general")

def human_review(state: AgentState, draft: str) -> dict:
    feedback = interrupt({
        "message": "Human review required. Enter approval or feedback to resume.",
        "category": state.get("category", "general"),
        "draft": draft,
    })
    feedback = str(feedback).strip() or "No additional feedback provided."
    return {
        "human_feedback": feedback,
        "final_response": f"{draft}\\n\\nHuman review: {feedback}",
    }

def respond_technical(state: AgentState) -> dict:
    draft = f"Technical support route selected for: {state['user_input']}. Please share relevant code or error details."
    return human_review(state, draft)

def respond_account(state: AgentState) -> dict:
    draft = f"Account support route selected for: {state['user_input']}. Never share passwords or one-time codes."
    return human_review(state, draft)

def respond_general(state: AgentState) -> dict:
    draft = f"General information route selected for: {state['user_input']}. I will provide a clear, concise response."
    return human_review(state, draft)

def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("classify", classify_node)
    builder.add_node("respond_technical", respond_technical)
    builder.add_node("respond_account", respond_account)
    builder.add_node("respond_general", respond_general)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_by_category,
        {
            "respond_technical": "respond_technical",
            "respond_account": "respond_account",
            "respond_general": "respond_general",
        },
    )
    for node in ("respond_technical", "respond_account", "respond_general"):
        builder.add_edge(node, END)
    return builder.compile(checkpointer=checkpointer)
