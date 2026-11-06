import pytest
from langgraph.checkpoint.memory import MemorySaver

from tests.test_classify import FIVE_INPUTS
from triage_agent.classify import Classification, RuleClassifier
from triage_agent.graph import build_graph
from triage_agent.responses import AUTO_REPLIES


def run(graph, message, thread="t"):
    config = {"configurable": {"thread_id": thread}}
    result = graph.invoke({"message": message, "trace": []}, config)
    return result, graph.get_state(config)


@pytest.fixture
def graph():
    return build_graph(RuleClassifier())


def test_graph_has_the_expected_nodes(graph):
    nodes = set(graph.get_graph().nodes)
    assert {"classify", "route", "human_review", "respond"} <= nodes


def test_default_graph_has_a_checkpointer(graph):
    assert graph.checkpointer is not None


# message -> (label, route, nodes visited before finishing or pausing)
EXPECTED = {
    "I was charged twice on my last invoice": ("billing", "auto", ["classify", "route", "respond"]),
    "The app crashes every time I open it": ("technical", "auto", ["classify", "route", "respond"]),
    "I want a refund, this product is broken": ("refund", "human", ["classify", "route"]),
    "Hello, what are your opening hours?": ("general", "auto", ["classify", "route", "respond"]),
    "asdf qwerty": ("unknown", "human", ["classify", "route"]),
}


def test_every_lesson_input_is_covered():
    assert set(EXPECTED) == {m for m, _ in FIVE_INPUTS}


@pytest.mark.parametrize("message", list(EXPECTED))
def test_routing_and_path_for_each_input(graph, message):
    label, route, trace = EXPECTED[message]
    _, state = run(graph, message)
    assert state.values["label"] == label
    assert state.values["route"] == route
    assert state.values["trace"] == trace


@pytest.mark.parametrize("message", [m for m, v in EXPECTED.items() if v[1] == "auto"])
def test_auto_route_finishes_without_pausing(graph, message):
    result, state = run(graph, message)
    assert "__interrupt__" not in result and state.next == ()
    assert result["response"] == AUTO_REPLIES[EXPECTED[message][0]]
    assert "human_review" not in state.values["trace"]


@pytest.mark.parametrize("message", [m for m, v in EXPECTED.items() if v[1] == "human"])
def test_human_route_pauses_at_human_review_with_no_response_yet(graph, message):
    result, state = run(graph, message)
    assert "__interrupt__" in result
    assert state.next == ("human_review",)
    assert "response" not in state.values   # nothing is sent while paused


class FixedClassifier:
    def __init__(self, label, confidence):
        self.result = Classification(label, confidence, "fixed")

    def classify(self, message):
        return self.result


def test_low_confidence_sends_a_safe_label_to_a_human():
    _, state = run(build_graph(FixedClassifier("billing", 0.2)), "x")
    assert state.values["route"] == "human"
    assert "unsure" in state.values["route_reason"]


def test_confidence_exactly_at_threshold_is_trusted():
    _, state = run(build_graph(FixedClassifier("billing", 0.6), confidence_threshold=0.6), "x")
    assert state.values["route"] == "auto"


def test_no_confidence_signal_is_not_treated_as_low_confidence():
    # An LLM classifier reports None; the router must not read that as 0.
    _, state = run(build_graph(FixedClassifier("technical", None)), "x")
    assert state.values["route"] == "auto"


def test_threshold_is_configurable():
    strict = build_graph(FixedClassifier("billing", 0.8), confidence_threshold=0.9)
    assert run(strict, "x")[1].values["route"] == "human"


def test_mixed_signal_message_goes_to_a_human(graph):
    _, state = run(graph, "I was charged twice and the app crashes")
    assert state.values["route"] == "human" and state.next == ("human_review",)
