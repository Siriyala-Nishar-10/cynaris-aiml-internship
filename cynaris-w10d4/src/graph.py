"""Three-node LangGraph workflow with conditional routing and human review."""
from typing import Annotated, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

class AgentState(TypedDict, total=False):
    user_input: str
    category: str
    route: str
    review_feedback: str
    response: str
    history: Annotated[list[str], list.__add__]

ROUTE_MAP = {
    "technical": "technical_support",
    "account": "account_support",
    "general": "general_support",
}

def classify(state: AgentState) -> dict:
    message = state.get("user_input", "").lower()
    if any(w in message for w in ("error", "bug", "python", "code", "install", "crash")):
        category = "technical"
    elif any(w in message for w in ("account", "login", "password", "billing", "payment")):
        category = "account"
    else:
        category = "general"
    return {"category": category, "route": None, "review_feedback": "", "response": ""}

def route(state: AgentState) -> dict:
    return {"route": ROUTE_MAP.get(state.get("category", "general"), "general_support")}

def route_condition(state: AgentState) -> str:
    return "respond" if state.get("category") in ROUTE_MAP else "end"

def respond(state: AgentState) -> dict:
    review = interrupt({
        "message": "Review this request before the agent responds.",
        "user_input": state.get("user_input", ""),
        "category": state.get("category", "general"),
        "route": state.get("route", "general_support"),
    })
    feedback = str(review).strip()
    if feedback.lower() in {"approved", "approve", "yes"}:
        response = (
            f"Request classified as '{state.get('category')}' and assigned to "
            f"'{state.get('route')}'. Human review approved the response."
        )
    else:
        response = f"Human review requested changes: {feedback}"
    entry = (
        f"User: {state.get('user_input', '')} | Category: {state.get('category')} | "
        f"Route: {state.get('route')} | Review: {feedback}"
    )
    return {
        "review_feedback": feedback,
        "response": response,
        "history": [entry],
    }

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("classify", classify)
    builder.add_node("route", route)
    builder.add_node("respond", respond)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "route")
    builder.add_conditional_edges("route", route_condition, {"respond": "respond", "end": END})
    return builder.compile(checkpointer=MemorySaver())

def default_config(thread_id: str = "w10d4-demo") -> dict:
    return {"configurable": {"thread_id": thread_id}}

def resume_after_review(graph, feedback: str, config: dict):
    return graph.invoke(Command(resume=feedback), config=config)
