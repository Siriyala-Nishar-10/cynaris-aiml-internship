"""
Automatic checks on a crew's output.

Three layers, from weakest to strongest evidence:

  output_metrics()   counts: words, URLs, domains, Sources heading. Descriptive only.
  quality_checks()   a small rubric on the final briefing (length, structure, sources,
                     and whether it cites URLs the research stage never mentioned).
  provenance()       whether every URL the Researcher cited was actually RETURNED by
                     the search tool during the run. This is what catches a model
                     inventing a plausible-looking link even when search was available.

All of these are crude. They cannot judge whether a claim is true or whether a real
URL supports the sentence next to it. That still needs a human reading the sources.
"""

import re
from typing import Iterable, Optional
from urllib.parse import urlparse, urlunparse

URL_PATTERN = re.compile(r"https?://[^\s)\]>\"']+")
SOURCES_HEADING = re.compile(r"^\s{0,3}#{0,6}\s*\**(sources|references)\**\s*:?\s*$", re.IGNORECASE | re.MULTILINE)
HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE)


def extract_urls(text: Optional[str]) -> list:
    """All http(s) URLs in the text, trailing punctuation stripped, order kept."""
    return [u.rstrip(".,;:") for u in URL_PATTERN.findall(text or "")]


def normalise_url(url: str) -> str:
    """Canonical form for comparing URLs: lowercase scheme+host, no fragment,
    no trailing slash, no leading 'www.'."""
    parts = urlparse(url.strip())
    host = parts.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    path = parts.path.rstrip("/")
    return urlunparse((parts.scheme.lower(), host, path, parts.params, parts.query, ""))


def domain_of(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def output_metrics(text: Optional[str]) -> dict:
    urls = extract_urls(text)
    return {
        "word_count": len((text or "").split()),
        "url_count": len(urls),
        "unique_urls": len(set(urls)),
        "unique_domains": len({domain_of(u) for u in urls}),
        "has_sources_section": bool(SOURCES_HEADING.search(text or "")),
        "says_no_sources": "no sources were consulted" in (text or "").lower(),
    }


def quality_checks(final_text: str, research_text: Optional[str] = None,
                   min_words: int = 150, max_words: int = 450, min_headings: int = 3) -> dict:
    """Rubric for the final briefing (target is ~300 words with a few headed sections).

    `unsupported_urls` are URLs cited in the final text that the research stage never
    mentioned: the Reviewer was told not to add any. It is None when no research text
    is available to compare against."""
    m = output_metrics(final_text)
    headings = len(HEADING_LINE.findall(final_text or ""))
    unsupported = None
    if research_text is not None:
        known = {normalise_url(u) for u in extract_urls(research_text)}
        unsupported = sorted({u for u in extract_urls(final_text) if normalise_url(u) not in known})

    checks = {
        "word_count": m["word_count"],
        "word_count_in_range": min_words <= m["word_count"] <= max_words,
        "heading_count": headings,
        "enough_headings": headings >= min_headings,
        "has_sources_section": m["has_sources_section"],
        "unsupported_urls": unsupported,
    }
    required = [checks["word_count_in_range"], checks["enough_headings"], checks["has_sources_section"]]
    if unsupported is not None:
        required.append(len(unsupported) == 0)
    checks["passed"] = all(required)
    return checks


def provenance(research_text: str, search_urls: Iterable[str]) -> dict:
    """Compare URLs cited by the Researcher with URLs the search tool actually returned."""
    returned = {normalise_url(u) for u in search_urls}
    cited = []
    for u in extract_urls(research_text):
        if normalise_url(u) not in {normalise_url(c) for c in cited}:
            cited.append(u)
    from_search = [u for u in cited if normalise_url(u) in returned]
    not_from_search = [u for u in cited if normalise_url(u) not in returned]
    return {
        "urls_returned_by_search": len(returned),
        "urls_cited": len(cited),
        "cited_and_returned": from_search,
        "cited_but_never_returned": not_from_search,
        "all_citations_traceable": len(not_from_search) == 0,
    }
