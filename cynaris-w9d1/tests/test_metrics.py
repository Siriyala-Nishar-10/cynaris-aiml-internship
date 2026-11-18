"""Tests for the output comparison measures."""
from research_crew.metrics import analyse, extract_urls, slugify, unique_domains

TEXT = ("Local models cut costs by 30% in 2025. See https://ollama.com/docs, "
        "https://www.example.com/a and https://ollama.com/docs again.")


def test_extract_urls_are_distinct_and_clean():
    assert extract_urls(TEXT) == ["https://ollama.com/docs", "https://www.example.com/a"]


def test_unique_domains_drop_www_and_repeats():
    urls = ["https://www.example.com/a", "https://example.com/b", "https://ollama.com/x"]
    assert unique_domains(urls) == ["example.com", "ollama.com"]


def test_analyse_counts_words_links_domains_and_figures():
    result = analyse(TEXT)
    assert result["urls"] == 2 and result["domains"] == 2
    assert result["figures"] >= 2                         # "30%" and "2025"
    assert result["words"] == len(TEXT.split())


def test_analyse_text_without_sources():
    assert analyse("Local models are useful.") == {"words": 4, "urls": 0, "domains": 0,
                                                   "figures": 0}


def test_slugify_is_filename_safe():
    assert slugify("AI & Ollama: Benefits?") == "ai-ollama-benefits"
    assert slugify("???") == "topic"
    assert len(slugify("word " * 30)) <= 40
