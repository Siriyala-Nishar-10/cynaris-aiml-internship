"""
Metrics and report for comparing a no-search run against a with-search run.

The metrics are crude proxies for "is the output grounded in sources":
  - URL count / unique domains: how many sources are cited
  - has_sources_section: whether a Sources/References heading exists
They CANNOT tell whether a URL is real or whether it supports the claim next to
it. A model without search can still print made-up URLs, and one with search can
misquote a real page. The report therefore ends with a section for manual checks.
"""

import re
from datetime import datetime
from urllib.parse import urlparse

URL_PATTERN = re.compile(r"https?://[^\s)\]>\"']+")
SOURCES_HEADING = re.compile(r"^\s{0,3}#{0,6}\s*\**(sources|references)\**\s*:?\s*$", re.IGNORECASE | re.MULTILINE)


def extract_urls(text: str) -> list:
    """All http(s) URLs in the text, trailing punctuation stripped, order kept."""
    return [u.rstrip(".,;:") for u in URL_PATTERN.findall(text or "")]


def domain_of(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def output_metrics(text: str) -> dict:
    urls = extract_urls(text)
    return {
        "word_count": len((text or "").split()),
        "url_count": len(urls),
        "unique_urls": len(set(urls)),
        "unique_domains": len({domain_of(u) for u in urls}),
        "has_sources_section": bool(SOURCES_HEADING.search(text or "")),
        "says_no_sources": "no sources were consulted" in (text or "").lower(),
    }


def build_report(topic: str, model: str, without: dict, with_: dict) -> str:
    """Markdown report. `without` / `with_` are dicts with 'metrics' and optional 'error'."""
    def cell(d, key):
        return "n/a (run failed)" if d.get("error") else str(d["metrics"][key])

    rows = [("Word count", "word_count"), ("URLs cited", "url_count"), ("Unique URLs", "unique_urls"),
            ("Unique domains", "unique_domains"), ("Has Sources section", "has_sources_section"),
            ("States 'no sources consulted'", "says_no_sources")]
    lines = [
        "# W9D2: Research Crew - Without vs With Web Search", "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"Topic: {topic}  ", f"Model: {model}", "",
        "> The metrics below are crude proxies. They cannot tell whether a URL is real or whether it",
        "> supports the claim beside it. Complete the manual checks at the end.", "",
        "## Metrics", "", "| Metric | Without search | With search |", "|---|---|---|",
        *[f"| {label} | {cell(without, key)} | {cell(with_, key)} |" for label, key in rows], "",
    ]
    for title, d in (("Without web search", without), ("With web search", with_)):
        lines += [f"## Output: {title}", ""]
        lines += [f"**Run failed:** {d['error']}", ""] if d.get("error") else [d["text"], ""]
    lines += [
        "## Manual checks (fill in after reading both outputs)", "",
        "- [ ] Opened at least 3 URLs from the with-search output: do they exist, and do they support the claim beside them?",
        "- [ ] Any URL in the without-search output? If yes, is it real, or invented?",
        "- [ ] Specific facts (numbers, names, dates) present in the with-search output but missing from the other:",
        "- [ ] Anything the Reviewer removed or flagged that you agree/disagree with:",
        "- [ ] Overall improvement from adding search, in my own words:", "",
    ]
    return "\n".join(lines)
