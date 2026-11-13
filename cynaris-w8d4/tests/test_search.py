"""Tests for the Haystack search service over the real sample documents."""
from dataclasses import replace

import pytest

pytest.importorskip("haystack")

from haystack import Document  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.search import (  # noqa: E402
    ModeUnavailable,
    NoDocumentsError,
    SearchService,
    chunk_documents,
    load_documents,
)


@pytest.fixture(scope="module")
def service():
    return SearchService(replace(load_settings({}), enable_dense=False))


def test_load_documents_reads_only_txt_files(tmp_path):
    (tmp_path / "a.txt").write_text("alpha beta", encoding="utf-8")
    (tmp_path / "b.md").write_text("ignored", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert [d.meta["source"] for d in docs] == ["a.txt"]


def test_empty_folder_raises_clear_error(tmp_path):
    with pytest.raises(NoDocumentsError):
        load_documents(tmp_path)


def test_chunking_splits_long_documents():
    doc = Document(content=" ".join(f"word{i}" for i in range(400)), meta={"source": "x.txt"})
    chunks = chunk_documents([doc], words=150, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.content.split()) <= 150 for c in chunks)
    assert all(c.meta["source"] == "x.txt" for c in chunks)


def test_service_indexes_all_sample_documents(service):
    assert service.n_documents == 5
    assert service.n_chunks >= 5
    assert service.modes() == ["bm25"]


@pytest.mark.parametrize("query, expected", [
    ("term frequency saturation and length normalisation", "bm25_retrieval.txt"),
    ("Railway PORT environment variable health check", "deploying_apis.txt"),
    ("reciprocal rank fusion merge two ranked lists", "hybrid_search.txt"),
])
def test_bm25_ranks_the_right_document_first(service, query, expected):
    assert service.search(query, top_k=1)[0]["source"] == expected


def test_top_k_is_respected_and_scores_are_descending(service):
    hits = service.search("retrieval", top_k=2)
    assert len(hits) <= 2
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert set(hits[0]) == {"text", "source", "score"}


def test_dense_mode_is_unavailable_when_disabled(service):
    with pytest.raises(ModeUnavailable):
        service.search("anything", mode="dense")


def test_unknown_mode_and_bad_top_k_are_rejected(service):
    with pytest.raises(ValueError):
        service.search("anything", mode="fuzzy")
    with pytest.raises(ValueError):
        service.search("anything", top_k=0)
