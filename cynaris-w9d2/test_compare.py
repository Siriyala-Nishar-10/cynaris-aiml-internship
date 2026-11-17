import unittest

from research_crew.compare import build_report, domain_of, extract_urls, output_metrics

SAMPLE = """# Briefing
RAG helps (see https://www.example.com/rag-guide). Also https://docs.other.org/page.html, and
a repeat of https://www.example.com/rag-guide.

## Sources
- https://www.example.com/rag-guide
- https://docs.other.org/page.html
"""


class TestExtraction(unittest.TestCase):
    def test_extracts_urls_and_strips_trailing_punctuation(self):
        urls = extract_urls(SAMPLE)
        self.assertEqual(urls[0], "https://www.example.com/rag-guide")        # ')' not included
        self.assertIn("https://docs.other.org/page.html", urls)               # ',' stripped
        self.assertEqual(len(urls), 5)

    def test_no_urls_in_plain_text(self):
        self.assertEqual(extract_urls("no links here"), [])
        self.assertEqual(extract_urls(None), [])

    def test_domain_strips_www_and_lowercases(self):
        self.assertEqual(domain_of("https://WWW.Example.com/x"), "example.com")
        self.assertEqual(domain_of("http://docs.other.org/a"), "docs.other.org")


class TestOutputMetrics(unittest.TestCase):
    def test_counts(self):
        m = output_metrics(SAMPLE)
        self.assertEqual(m["url_count"], 5)
        self.assertEqual(m["unique_urls"], 2)
        self.assertEqual(m["unique_domains"], 2)
        self.assertTrue(m["has_sources_section"])
        self.assertFalse(m["says_no_sources"])

    def test_sources_heading_variants(self):
        for heading in ("## Sources", "Sources:", "**References**", "# References"):
            self.assertTrue(output_metrics(f"text\n{heading}\n- a")["has_sources_section"], heading)
        self.assertFalse(output_metrics("We discuss sources of error in the text.")["has_sources_section"])

    def test_no_sources_statement_detected(self):
        m = output_metrics("Summary.\n\nSources: No sources were consulted.")
        self.assertTrue(m["says_no_sources"])
        self.assertEqual(m["url_count"], 0)

    def test_empty_text(self):
        m = output_metrics("")
        self.assertEqual((m["word_count"], m["url_count"]), (0, 0))


class TestReport(unittest.TestCase):
    def test_report_contains_tables_outputs_and_manual_checklist(self):
        without = {"text": "baseline text. No sources were consulted.", "metrics": output_metrics("baseline text. No sources were consulted.")}
        with_ = {"text": SAMPLE, "metrics": output_metrics(SAMPLE)}
        report = build_report("RAG", "ollama/llama3.2:3b", without, with_)
        self.assertIn("| URLs cited | 0 | 5 |", report)
        self.assertIn("## Output: Without web search", report)
        self.assertIn("## Output: With web search", report)
        self.assertIn("## Manual checks", report)
        self.assertIn("crude proxies", report)

    def test_failed_run_is_reported_not_hidden(self):
        failed = {"error": "TimeoutError: model too slow", "metrics": output_metrics("")}
        ok = {"text": SAMPLE, "metrics": output_metrics(SAMPLE)}
        report = build_report("RAG", "m", failed, ok)
        self.assertIn("**Run failed:** TimeoutError", report)
        self.assertIn("n/a (run failed)", report)


if __name__ == "__main__":
    unittest.main()
