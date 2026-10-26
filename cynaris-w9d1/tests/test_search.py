"""Tests for the web search helpers, using a fake search function (no network)."""
from research_crew.search import clean_for_prompt, format_results, run_search

HITS = [
    {"title": "Ollama docs", "url": "https://ollama.com/docs", "snippet": "Run models locally."},
    {"title": "Blog", "url": "https://example.com/post", "snippet": "A review."},
]


def test_format_results_numbers_and_includes_urls():
    text = format_results(HITS)
    assert text.startswith("[1] Ollama docs\nURL: https://ollama.com/docs\nRun models locally.")
    assert "[2] Blog" in text


def test_format_results_handles_empty_list():
    assert format_results([]) == "No web results found."


def test_run_search_returns_formatted_results():
    seen = {}

    def fake(query, max_results):
        seen["args"] = (query, max_results)
        return HITS

    text = run_search("  ollama   news ", max_results=3, search_fn=fake)
    assert "https://ollama.com/docs" in text
    assert seen["args"] == ("ollama news", 3)            # whitespace tidied


def test_run_search_reports_failures_instead_of_raising():
    def broken(query, max_results):
        raise ConnectionError("offline")

    text = run_search("anything", search_fn=broken)
    assert "failed" in text and "ConnectionError" in text


def test_run_search_rejects_empty_query():
    assert run_search("   ", search_fn=lambda q, n: HITS) == "Empty search query."


def test_clean_for_prompt_removes_braces():
    assert clean_for_prompt('{"a": 1} {topic}') == '("a": 1) (topic)'
