"""LangGraph layer: retrieve -> (route) -> research crew | fallback."""
from __future__ import annotations

from typing import TypedDict

from .config import Settings
from .crew import run_crew
from .memory import VectorMemory, embed

FALLBACK = "I couldn't find anything relevant in the indexed documents."


class State(TypedDict, total=False):
    question: str
    contexts: list[str]
    sources: list[str]
    answer: str


def route_after_retrieval(state: State) -> str:
    """Send the run to the crew only if retrieval found something."""
    return "research" if state.get("contexts") else "fallback"


def build_graph(cfg: Settings, memory: VectorMemory):
    from langgraph.graph import END, StateGraph

    def retrieve(state: State) -> State:
        hits = memory.search(embed(state["question"], cfg), cfg.top_k)
        return {"contexts": [h.text for h in hits], "sources": [h.source for h in hits]}

    def research(state: State) -> State:
        return {"answer": run_crew(state["question"], state["contexts"], cfg)}

    def fallback(state: State) -> State:
        return {"answer": FALLBACK}

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("research", research)
    g.add_node("fallback", fallback)
    g.set_entry_point("retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieval,
                            {"research": "research", "fallback": "fallback"})
    g.add_edge("research", END)
    g.add_edge("fallback", END)
    return g.compile()
