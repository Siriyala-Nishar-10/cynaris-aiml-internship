import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from triage_agent.classify import Classification, RuleClassifier
from triage_agent.graph import build_graph, normalise_decision

REFUND = "I want a refund, this product is broken"


def start(graph, message, thread="t"):
    config = {"configurable": {"thread_id": thread}}
    result = graph.invoke({"message": message, "trace": []}, config)
    return config, result


def resume(graph, config, value):
    return graph.invoke(Command(resume=value), config)


def test_pause_shows_the_reviewer_what_they_need():
    _, result = start(build_graph(RuleClassifier()), REFUND)
    payload = result["__interrupt__"][0].value
    assert payload["message"] == REFUND
    assert payload["label"] == "refund"
    assert "human approval" in payload["reason"]


def test_approval_resumes_and_completes_the_full_path():
    graph = build_graph(RuleClassifier())
    config, _ = start(graph, REFUND)
    result = resume(graph, config, {"approved": True, "note": "within 30 days"})
    state = graph.get_state(config)
    assert "approved your refund" in result["response"] and "within 30 days" in result["response"]
    assert state.values["trace"] == ["classify", "route", "human_review", "respond"]
    assert state.next == ()          # nothing left to run


def test_denial_produces_a_decline_with_the_reason():
    graph = build_graph(RuleClassifier())
    config, _ = start(graph, REFUND)
    result = resume(graph, config, {"approved": False, "note": "outside return window"})
    assert "did not approve" in result["response"]
    assert "outside return window" in result["response"]


def test_unknown_message_uses_the_humans_note_as_the_reply():
    graph = build_graph(RuleClassifier())
    config, _ = start(graph, "asdf qwerty")
    result = resume(graph, config, {"approved": True, "note": "Could you tell us more?"})
    assert result["response"] == "Could you tell us more?"


@pytest.mark.parametrize("ambiguous", ["yes", "ok", None, 1, {"approved": "yes"}, {}])
def test_ambiguous_human_input_never_counts_as_approval(ambiguous):
    assert normalise_decision(ambiguous)["approved"] is False


def test_explicit_approval_forms_are_accepted():
    assert normalise_decision(True)["approved"] is True
    assert normalise_decision({"approved": True})["approved"] is True


def test_resuming_with_a_typo_declines_rather_than_refunds():
    graph = build_graph(RuleClassifier())
    config, _ = start(graph, REFUND)
    assert "did not approve" in resume(graph, config, "yes")["response"]


def test_two_paused_tickets_are_independent():
    graph = build_graph(RuleClassifier())
    cfg_a, _ = start(graph, REFUND, "ticket-A")
    cfg_b, _ = start(graph, REFUND, "ticket-B")
    assert graph.get_state(cfg_a).next == graph.get_state(cfg_b).next == ("human_review",)

    resume(graph, cfg_a, {"approved": True, "note": "A ok"})
    assert graph.get_state(cfg_a).next == ()
    assert graph.get_state(cfg_b).next == ("human_review",)   # B still waiting

    result_b = resume(graph, cfg_b, {"approved": False, "note": "B no"})
    assert "A ok" not in result_b["response"] and "B no" in result_b["response"]


class CountingClassifier(RuleClassifier):
    calls = 0

    def classify(self, message):
        CountingClassifier.calls += 1
        return super().classify(message)


def test_resume_does_not_reclassify_the_message():
    # Matters for LLM classifiers: re-running them would cost time and could
    # change the label mid-conversation.
    CountingClassifier.calls = 0
    graph = build_graph(CountingClassifier())
    config, _ = start(graph, REFUND)
    resume(graph, config, {"approved": True, "note": ""})
    assert CountingClassifier.calls == 1


def test_library_behaviour_resume_needs_a_checkpointer():
    """Documents (and guards) the trap behind build_graph's safe default:
    without a checkpointer a run appears to pause, but can never be resumed."""

    class S(TypedDict, total=False):
        out: str

    def ask(state):
        return {"out": str(interrupt("approve?"))}

    g = StateGraph(S)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    app = g.compile()                                   # no checkpointer
    cfg = {"configurable": {"thread_id": "x"}}
    assert "__interrupt__" in app.invoke({}, cfg)       # looks paused...
    with pytest.raises(RuntimeError, match="checkpointer"):
        app.invoke(Command(resume="yes"), cfg)          # ...but cannot resume


@pytest.mark.parametrize("note", ["within 30 days", "within 30 days."])
def test_reply_sentences_are_cleanly_punctuated(note):
    graph = build_graph(RuleClassifier())
    config, _ = start(graph, REFUND)
    reply = resume(graph, config, {"approved": True, "note": note})["response"]
    assert reply == ("A human agent approved your refund request. "
                     "Note: within 30 days. It will now be processed.")
