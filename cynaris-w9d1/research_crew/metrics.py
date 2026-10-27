"""Simple, transparent measures for comparing two crew outputs.

These are rough proxies (links, domains, figures). They support a manual quality review;
they do not replace it.
"""
import re
from urllib.parse import urlparse

URL_RE = re.compile(r"https?://[^\s)\]>\"'*]+")
FIGURE_RE = re.compile(r"\b\d[\d,.]*%?")


def slugify(text: str, max_len: int = 40) -> str:
    """File-name-safe version of a topic."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "topic"


def extract_urls(text: str) -> list[str]:
    """Every distinct URL in the text, with trailing punctuation removed."""
    return list(dict.fromkeys(u.rstrip(".,;:") for u in URL_RE.findall(text)))


def unique_domains(urls: list[str]) -> list[str]:
    """Distinct web domains (without 'www.') for a list of URLs."""
    domains = (urlparse(u).netloc.lower().removeprefix("www.") for u in urls)
    return list(dict.fromkeys(d for d in domains if d))


def analyse(text: str) -> dict:
    """Counts that show how well-sourced and concrete a text is."""
    urls = extract_urls(text)
    return {"words": len(text.split()), "urls": len(urls),
            "domains": len(unique_domains(urls)), "figures": len(FIGURE_RE.findall(text))}
