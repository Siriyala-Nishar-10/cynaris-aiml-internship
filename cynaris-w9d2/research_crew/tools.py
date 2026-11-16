"""
Web search for the Researcher agent.

Layout:
  web_search()          pure function: query -> list of {title, url, snippet}
  format_results()      list -> numbered text block the LLM reads
  build_search_tool()   wraps the two above as a CrewAI tool

Splitting it this way means everything except the thin CrewAI wrapper can be unit
tested without CrewAI installed or any network access.

Backend: DuckDuckGo via the `ddgs` package (no API key needed). The older package
name `duckduckgo_search` is tried as a fallback. If a keyed service such as Serper
is preferred, only `_default_backend` needs to change.
"""

from typing import Callable, List, Optional

SNIPPET_LIMIT = 300


class SearchError(RuntimeError):
    """Search could not be performed (missing package, network, rate limit...)."""


def _default_backend(query: str, max_results: int) -> List[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:
            raise SearchError("No search package installed. Run: pip install ddgs") from exc
    try:
        return list(DDGS().text(query, max_results=max_results))
    except Exception as exc:  # network errors, rate limits and library errors all land here
        raise SearchError(f"Search request failed: {exc}") from exc


def web_search(query: str, max_results: int = 5,
               backend: Optional[Callable[[str, int], List[dict]]] = None) -> List[dict]:
    """Return up to `max_results` unique results as {title, url, snippet}.

    `backend` can be injected for testing; it must return dicts using the
    DuckDuckGo keys ('title', 'href', 'body')."""
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if max_results < 1:
        raise ValueError("max_results must be at least 1")

    raw = (backend or _default_backend)(query.strip(), max_results)
    results, seen = [], set()
    for item in raw:
        url = (item.get("href") or item.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        snippet = " ".join((item.get("body") or item.get("snippet") or "").split())
        results.append({
            "title": " ".join((item.get("title") or "").split()) or url,
            "url": url,
            "snippet": snippet[:SNIPPET_LIMIT],
        })
        if len(results) == max_results:
            break
    return results


def format_results(results: List[dict]) -> str:
    """Numbered, citation-friendly text for the agent to read."""
    if not results:
        return "No results found. Try a different or broader query."
    blocks = [f"[{i}] {r['title']}\nURL: {r['url']}\n{r['snippet']}" for i, r in enumerate(results, 1)]
    return "\n\n".join(blocks)


def search_for_agent(query: str, max_results: int = 5, backend=None) -> str:
    """What the agent actually calls. Failures become readable text instead of
    exceptions, so one bad search does not crash the whole crew run."""
    try:
        return format_results(web_search(query, max_results, backend))
    except (SearchError, ValueError) as exc:
        return f"SEARCH ERROR: {exc}. Continue using only information you are certain about, and say so."


def build_search_tool(max_results: int = 5):
    """CrewAI tool wrapping `search_for_agent`. crewai is imported here, not at
    module level, so the rest of this module works without it installed."""
    from crewai.tools import tool

    @tool("Web Search")
    def web_search_tool(query: str) -> str:
        """Search the web and return numbered results with titles, URLs and snippets.
        Use short, specific queries. Cite only URLs that appear in the results."""
        return search_for_agent(query, max_results)

    return web_search_tool
