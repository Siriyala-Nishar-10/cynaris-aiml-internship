import sys
import types
import unittest
from unittest import mock

from research_crew import tools
from research_crew.tools import (SearchError, format_results, search_for_agent, web_search)


def fake_backend(items):
    return lambda query, n: items


RAW = [
    {"title": "  Alpha\n page ", "href": "https://a.example/1", "body": "First   snippet\nwith   spaces"},
    {"title": "Alpha duplicate", "href": "https://a.example/1", "body": "same url again"},
    {"title": "Beta", "href": "https://b.example/2", "body": "x" * 1000},
    {"title": "No url", "href": "", "body": "dropped"},
    {"title": "Gamma", "url": "https://c.example/3", "snippet": "uses the url/snippet keys"},
]


class TestWebSearch(unittest.TestCase):
    def test_normalises_dedupes_and_drops_empty_urls(self):
        out = web_search("q", max_results=10, backend=fake_backend(RAW))
        self.assertEqual([r["url"] for r in out],
                         ["https://a.example/1", "https://b.example/2", "https://c.example/3"])
        self.assertEqual(out[0]["title"], "Alpha page")                 # whitespace collapsed
        self.assertEqual(out[0]["snippet"], "First snippet with spaces")

    def test_snippet_is_truncated(self):
        out = web_search("q", 10, fake_backend(RAW))
        self.assertEqual(len(out[1]["snippet"]), tools.SNIPPET_LIMIT)

    def test_max_results_is_respected(self):
        self.assertEqual(len(web_search("q", max_results=2, backend=fake_backend(RAW))), 2)

    def test_missing_title_falls_back_to_url(self):
        out = web_search("q", 5, fake_backend([{"href": "https://z.example", "body": "b"}]))
        self.assertEqual(out[0]["title"], "https://z.example")

    def test_blank_query_and_bad_limit_rejected(self):
        with self.assertRaises(ValueError):
            web_search("   ", backend=fake_backend([]))
        with self.assertRaises(ValueError):
            web_search("q", max_results=0, backend=fake_backend([]))

    def test_query_is_stripped_before_reaching_backend(self):
        seen = []
        web_search("  hello  ", 3, lambda q, n: seen.append((q, n)) or [])
        self.assertEqual(seen, [("hello", 3)])


class TestFormatting(unittest.TestCase):
    def test_numbered_blocks_with_urls(self):
        text = format_results(web_search("q", 5, fake_backend(RAW)))
        self.assertIn("[1] Alpha page\nURL: https://a.example/1", text)
        self.assertIn("[3] Gamma", text)

    def test_empty_results_message(self):
        self.assertIn("No results found", format_results([]))


class TestSearchForAgent(unittest.TestCase):
    def test_success_returns_formatted_text(self):
        self.assertIn("URL: https://a.example/1", search_for_agent("q", 5, fake_backend(RAW)))

    def test_backend_failure_becomes_text_not_exception(self):
        def boom(q, n):
            raise SearchError("rate limited")
        out = search_for_agent("q", 5, boom)
        self.assertTrue(out.startswith("SEARCH ERROR"))
        self.assertIn("rate limited", out)

    def test_invalid_query_becomes_text_not_exception(self):
        self.assertTrue(search_for_agent("   ", 5, fake_backend([])).startswith("SEARCH ERROR"))


class TestDefaultBackend(unittest.TestCase):
    def test_missing_package_raises_actionable_error(self):
        with mock.patch.dict(sys.modules, {"ddgs": None, "duckduckgo_search": None}):
            with self.assertRaises(SearchError) as ctx:
                tools._default_backend("q", 3)
        self.assertIn("pip install ddgs", str(ctx.exception))

    def test_uses_ddgs_when_available(self):
        fake = types.ModuleType("ddgs")
        class DDGS:
            def text(self, query, max_results):
                return [{"title": "T", "href": "https://x.example", "body": "b"}]
        fake.DDGS = DDGS
        with mock.patch.dict(sys.modules, {"ddgs": fake}):
            out = tools._default_backend("q", 3)
        self.assertEqual(out[0]["href"], "https://x.example")

    def test_library_errors_are_wrapped(self):
        fake = types.ModuleType("ddgs")
        class DDGS:
            def text(self, query, max_results):
                raise RuntimeError("202 Ratelimit")
        fake.DDGS = DDGS
        with mock.patch.dict(sys.modules, {"ddgs": fake}):
            with self.assertRaises(SearchError):
                tools._default_backend("q", 3)


if __name__ == "__main__":
    unittest.main()
