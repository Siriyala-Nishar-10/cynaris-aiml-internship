"""LangGraph workflow with conditional routing, human review, and conversation memory."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ConversationState(TypedDict, total=False):
    user_input: str
    category: str
    route: str | None
    draft_response: str
    human_feedback: str
    final_response: str
    conversation_history: Annotated[list[str], list.__add__]


def classify(state: ConversationState) -> dict:
    """Classify the user message and clear any previous route."""

    message = state.get("user_input", "").lower()

    technical_terms = (
        "error", "bug", "crash", "code",
        "python", "install", "not working"
    )

    account_terms = (
        "login", "log in", "password",
        "account", "sign in", "reset"
    )

    if any(term in message for term in technical_terms):
        category = "technical"
    elif any(term in message for term in account_terms):
        category = "account"
    else:
        category = "general"

    # Clear the previous route to prevent stale checkpoint data.
    return {
        "category": category,
        "route": None
    }


def route(state: ConversationState) -> dict:
    """Select a support route."""

    category = state.get("category", "general")

    route_name = {
        "technical": "technical_support",
        "account": "account_support",
        "general": "general_support",
    }.get(category, "general_support")

    return {"route": route_name}


def should_use_specialist_route(state: ConversationState) -> str:
    """Choose whether the route node is required."""

    if state.get("category") in {"technical", "account"}:
        return "route"

    return "respond"


def after_route(state: ConversationState) -> str:
    """Continue to the response node."""

    return "respond"


def respond(state: ConversationState) -> dict:
    """Generate a response after human review."""

    category = state.get("category", "general")
    route_name = state.get("route") or "general_support"
    user_input = state.get("user_input", "")

    draft = (
        f"Request category: {category}. "
        f"Selected route: {route_name}. "
        f"Please review the request: {user_input}"
    )

    # Pause execution for human review.
    feedback = interrupt({
        "type": "human_review",
        "draft_response": draft,
        "prompt": "Approve this draft or provide feedback."
    })

    feedback_text = str(feedback)

    final = f"{draft} Human review: {feedback_text}"

    history_entry = (
        f"User: {user_input}\n"
        f"Assistant: {final}"
    )

    return {
        "route": route_name,
        "draft_response": draft,
        "human_feedback": feedback_text,
        "final_response": final,
        "conversation_history": [history_entry]
    }


def build_graph(checkpointer=None):
    """Build and compile the three-node LangGraph."""

    builder = StateGraph(ConversationState)

    builder.add_node("classify", classify)
    builder.add_node("route", route)
    builder.add_node("respond", respond)

    builder.add_edge(START, "classify")

    builder.add_conditional_edges(
        "classify",
        should_use_specialist_route,
        {
            "route": "route",
            "respond": "respond"
        }
    )

    builder.add_conditional_edges(
        "route",
        after_route,
        {
            "respond": "respond"
        }
    )

    builder.add_edge("respond", END)

    return builder.compile(
        checkpointer=checkpointer or MemorySaver()
    )


def default_config(thread_id: str = "w10d3-demo") -> dict:
    """Create a conversation configuration."""

    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


def resume_after_review(graph, feedback: str, config: dict):
    """Resume execution after human review."""

    return graph.invoke(
        Command(resume=feedback),
        config=config
    )