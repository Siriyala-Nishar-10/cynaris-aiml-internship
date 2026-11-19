import pytest
import requests

from triage_agent.classify import RuleClassifier
from triage_agent.cli import main, process_message
from triage_agent.graph import build_graph
from triage_agent.llm_client import normalise_host, ollama_chat_fn


def test_process_message_pauses_asks_human_then_resumes():
    asked = []

    def human(payload):
        asked.append(payload)
        return {"approved": True, "note": "ok"}

    result = process_message(build_graph(RuleClassifier()), "I want a refund", "t1", human)
    assert len(asked) == 1 and asked[0]["label"] == "refund"
    assert result["trace"][-2:] == ["human_review", "respond"]


def test_process_message_never_asks_a_human_for_routine_tickets():
    def human(payload):
        raise AssertionError("human should not be consulted")

    result = process_message(build_graph(RuleClassifier()),
                             "The app crashes on launch", "t2", human)
    assert result["trace"] == ["classify", "route", "respond"]


def test_demo_prints_all_five_inputs_and_shows_where_it_paused(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert out.count("[PAUSED at human_review]") == 2      # refund + unknown
    for text in ("charged twice", "crashes every time", "opening hours", "asdf qwerty"):
        assert text in out


def test_diagram_command_writes_mermaid(tmp_path, capsys):
    target = tmp_path / "g.mmd"
    assert main(["diagram", "--out", str(target)]) == 0
    text = target.read_text()
    assert "classify" in text and "human_review" in text


# ---- Ollama client (HTTP faked) ----

class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, outcome):
        self.outcome = outcome

    def post(self, url, json, timeout):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_chat_fn_returns_model_text():
    session = FakeSession(FakeResponse(body={"message": {"content": "billing"}}))
    assert ollama_chat_fn("m", "http://x", session=session)("hi") == "billing"


def test_chat_fn_explains_how_to_start_ollama():
    chat = ollama_chat_fn("m", "http://x", session=FakeSession(requests.exceptions.ConnectionError()))
    with pytest.raises(RuntimeError, match="ollama serve"):
        chat("hi")


def test_chat_fn_explains_how_to_pull_a_missing_model():
    chat = ollama_chat_fn("llama3.2:3b", "http://x", session=FakeSession(FakeResponse(status=404)))
    with pytest.raises(RuntimeError, match="ollama pull llama3.2:3b"):
        chat("hi")


@pytest.mark.parametrize("given,expected", [
    ("127.0.0.1:11434", "http://127.0.0.1:11434"),
    ("http://localhost:11434", "http://localhost:11434"),
])
def test_host_without_scheme_gets_one(given, expected):
    assert normalise_host(given) == expected
