"""The LangGraph state machine.

    START -> classify -> route --(auto)---------> respond -> END
                           |
                           +--(human)--> human_review --> respond -> END
                                         (pauses via interrupt())
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .classify import Classifier
from .responses import compose_response
from .state import AgentState

# Labels that ALWAYS need a person: money movement, and "we couldn't tell".
HUMAN_LABELS = {"refund", "unknown"}


def normalise_decision(answer) -> dict:
    """Fail safe: only an explicit approval counts as approval.

    Anything ambiguous ("yes", "ok", None, {"approved": "yes"}) is treated as a
    denial, so a typo from the reviewer can never trigger a refund.
    """
    if isinstance(answer, dict):
        return {"approved": answer.get("approved") is True,
                "note": str(answer.get("note") or "")}
    if isinstance(answer, bool):
        return {"approved": answer, "note": ""}
    return {"approved": False, "note": ""}


def build_graph(classifier: Classifier, confidence_threshold: float = 0.6,
                checkpointer=None):
    def classify(state: AgentState) -> AgentState:
        result = classifier.classify(state["message"])
        return {"label": result.label, "confidence": result.confidence,
                "method": result.method, "trace": ["classify"]}

    def route(state: AgentState) -> AgentState:
        label, confidence = state["label"], state.get("confidence")
        if label == "refund":
            reason = "refund requests always need human approval"
        elif label == "unknown":
            reason = "no category matched this message"
        elif confidence is not None and confidence < confidence_threshold:
            reason = f"classifier unsure (confidence {confidence:.2f} < {confidence_threshold})"
        else:
            reason = ""
        return {"route": "human" if reason else "auto",
                "route_reason": reason or "confident, low-risk", "trace": ["route"]}

    def pick_next(state: AgentState) -> str:
        """The conditional edge: its return value chooses the next node."""
        return "human_review" if state["route"] == "human" else "respond"

    def human_review(state: AgentState) -> AgentState:
        # On resume LangGraph re-runs this node from its first line, so anything
        # above interrupt() executes twice. Keep it free of side effects.
        answer = interrupt({"message": state["message"], "label": state["label"],
                            "confidence": state.get("confidence"),
                            "reason": state["route_reason"]})
        return {"human_decision": normalise_decision(answer), "trace": ["human_review"]}

    def respond(state: AgentState) -> AgentState:
        return {"response": compose_response(state["label"], state.get("human_decision")),
                "trace": ["respond"]}

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("route", route)
    graph.add_node("human_review", human_review)
    graph.add_node("respond", respond)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "route")
    graph.add_conditional_edges("route", pick_next,
                                {"respond": "respond", "human_review": "human_review"})
    graph.add_edge("human_review", "respond")
    graph.add_edge("respond", END)

    # A checkpointer is mandatory for pause/resume. Without one, interrupt()
    # still "pauses" but the run can never be resumed (RuntimeError), so the
    # safe default is an in-memory one. Use a persistent saver in production.
    return graph.compile(checkpointer=checkpointer or MemorySaver())
