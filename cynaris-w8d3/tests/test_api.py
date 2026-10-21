"""API tests (fake service) plus one real BM25 test over the sample documents."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.search import ModeUnavailable


class FakeService:
    n_documents, n_chunks = 2, 3

    def modes(self):
        return ["bm25"]

    def search(self, query, mode="bm25", top_k=3):
        if mode == "dense":
            raise ModeUnavailable("dense mode is not enabled")
        return [{"text": f"result for {query}", "source": "fake.txt", "score": 1.5}][:top_k]


def client():
    return TestClient(create_app(FakeService()))


def test_health_ok():
    assert client().get("/health").json() == {"status": "ok"}


def test_info_reports_documents_and_modes():
    body = client().get("/info").json()
    assert body == {"documents": 2, "chunks": 3, "modes": ["bm25"]}


def test_search_returns_hits():
    r = client().get("/search", params={"q": "railway deploy"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "bm25" and body["results"][0]["source"] == "fake.txt"


def test_dense_unavailable_returns_501():
    assert client().get("/search", params={"q": "anything", "mode": "dense"}).status_code == 501


def test_bad_mode_and_short_query_are_rejected():
    assert client().get("/search", params={"q": "valid query", "mode": "nope"}).status_code == 422
    assert client().get("/search", params={"q": "a"}).status_code == 422
    assert client().get("/search", params={"q": "valid query", "top_k": 99}).status_code == 422


def test_real_bm25_finds_the_right_sample_document():
    pytest.importorskip("haystack")
    from app.search import SearchService
    svc = SearchService(docs_dir="sample_docs", enable_dense=False)
    top = svc.search("term frequency saturation and length normalisation", top_k=1)[0]
    assert top["source"] == "bm25_retrieval.txt"
    top = svc.search("Railway PORT environment variable health check", top_k=1)[0]
    assert top["source"] == "deploying_apis.txt"
