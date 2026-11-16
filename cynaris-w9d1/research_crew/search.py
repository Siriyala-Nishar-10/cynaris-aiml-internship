"""Web search helpers. Pure Python: no CrewAI import, so they are easy to test.

Uses DuckDuckGo through the `ddgs` package, which needs no API key.
"""
import re


def search_ddg(query: str, max_results: int = 5) -> list[dict]:
    """Return [{"title", "url", "snippet"}] from DuckDuckGo."""
    try:
        from ddgs import DDGS
    except ImportError:                              # older package name
        from duckduckgo_search import DDGS
    hits = DDGS().text(query, max_results=max_results) or []
    return [{"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")}
            for h in hits]


def format_results(results: list[dict]) -> str:
    """Numbered plain-text list the agent can read and cite."""
    if not results:
        return "No web results found."
    blocks = [f"[{i}] {r['title']}\nURL: {r['url']}\n{r['snippet']}"
              for i, r in enumerate(results, 1)]
    return "\n\n".join(blocks)


def clean_for_prompt(text: str) -> str:
    """Remove curly braces: CrewAI treats {name} in task text as a template placeholder."""
    return text.replace("{", "(").replace("}", ")")


def run_search(query: str, max_results: int = 5, search_fn=search_ddg) -> str:
    """Search and return formatted text. Failures come back as a message, so an agent can
    carry on (or say the search failed) instead of the whole crew crashing."""
    query = re.sub(r"\s+", " ", query).strip()
    if not query:
        return "Empty search query."
    try:
        return format_results(search_fn(query, max_results))
    except Exception as exc:                         # network errors, rate limits, ...
        return f"Web search failed ({type(exc).__name__}). Continue without web results."
