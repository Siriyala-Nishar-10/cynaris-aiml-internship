"""API tests with a fake service, so they run fast and need no Haystack index."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.search import ModeUnavailable


class FakeService:
    n_documents, n_chunks = 2, 3

    def __init__(self, results=None, error=None):
        self.results = [{"text": "a chunk", "source": "fake.txt", "score": 1.5}] \
            if results is None else results
        self.error = error
        self.calls = []

    def modes(self):
        return ["bm25"]

    def search(self, query, mode="bm25", top_k=3):
        self.calls.append((query, mode, top_k))
        if self.error:
            raise self.error
        if mode == "dense":
            raise ModeUnavailable("dense mode is not enabled")
        return self.results[:top_k]


def make_client(service=None):
    service = service or FakeService()
    return TestClient(create_app(service)), service


def test_health_ok():
    client, _ = make_client()
    assert client.get("/health").json() == {"status": "ok"}


def test_info_reports_documents_chunks_and_modes():
    client, _ = make_client()
    assert client.get("/info").json() == {"documents": 2, "chunks": 3, "modes": ["bm25"]}


def test_search_returns_hits_with_latency():
    client, _ = make_client()
    body = client.get("/search", params={"q": "railway deploy"}).json()
    assert body["mode"] == "bm25"
    assert body["results"][0] == {"text": "a chunk", "source": "fake.txt", "score": 1.5}
    assert body["latency_s"] >= 0


def test_query_is_stripped_before_searching():
    client, service = make_client()
    client.get("/search", params={"q": "  railway deploy  "})
    assert service.calls == [("railway deploy", "bm25", 3)]


def test_no_matches_returns_empty_list_not_an_error():
    client, _ = make_client(FakeService(results=[]))
    r = client.get("/search", params={"q": "zzzz qqqq"})
    assert r.status_code == 200 and r.json()["results"] == []


def test_top_k_is_forwarded():
    client, service = make_client()
    client.get("/search", params={"q": "railway", "top_k": 5})
    assert service.calls[0][2] == 5


@pytest.mark.parametrize("params", [
    {"q": "a"},                                  # too short
    {"q": "   "},                                # only spaces: passes length check, fails strip
    {"q": "x" * 301},                            # too long
    {"q": "valid query", "mode": "fuzzy"},       # unknown mode
    {"q": "valid query", "top_k": 0},
    {"q": "valid query", "top_k": 11},
])
def test_invalid_requests_are_rejected_with_422(params):
    client, service = make_client()
    assert client.get("/search", params=params).status_code == 422
    assert service.calls == []                   # the service is never reached


def test_missing_query_is_rejected():
    client, _ = make_client()
    assert client.get("/search").status_code == 422


def test_dense_mode_returns_501_when_disabled():
    client, _ = make_client()
    r = client.get("/search", params={"q": "anything", "mode": "dense"})
    assert r.status_code == 501
    assert "not enabled" in r.json()["detail"]


def test_unexpected_error_returns_clean_500():
    client, _ = make_client(FakeService(error=RuntimeError("index corrupted: secret path")))
    r = client.get("/search", params={"q": "railway"})
    assert r.status_code == 500
    assert r.json() == {"detail": "search failed"}   # internal message is not leaked
