"""API tests. They use the stub backend, so they need no Ollama and run fast in CI."""
from fastapi.testclient import TestClient

from app.main import create_app
from app.rag import RagResult, StubBackend


class BrokenBackend:
    def query(self, question):
        raise RuntimeError("boom")


def client(backend):
    return TestClient(create_app(backend))


def test_health_ok():
    r = client(StubBackend()).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query_returns_answer_and_sources():
    r = client(StubBackend()).post("/query", json={"question": "What is RAG?"})
    assert r.status_code == 200
    body = r.json()
    assert "What is RAG?" in body["answer"]
    assert body["sources"] == ["stub.txt"]
    assert body["latency_s"] >= 0


def test_short_question_is_rejected():
    r = client(StubBackend()).post("/query", json={"question": "hi"})
    assert r.status_code == 422


def test_backend_failure_returns_500_and_counts_error():
    c = client(BrokenBackend())
    assert c.post("/query", json={"question": "anything here"}).status_code == 500
    stats = c.get("/stats").json()
    assert stats["requests"] == 1 and stats["errors"] == 1


def test_stats_track_requests():
    c = client(StubBackend())
    c.post("/query", json={"question": "first question"})
    c.post("/query", json={"question": "second question"})
    assert c.get("/stats").json()["requests"] == 2


def test_rag_result_defaults():
    assert RagResult(answer="x").sources == []
