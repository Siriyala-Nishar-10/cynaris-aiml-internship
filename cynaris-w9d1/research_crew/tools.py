"""The web search tool the Researcher agent can call (CrewAI BaseTool wrapper)."""
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from research_crew.search import run_search


class SearchInput(BaseModel):
    query: str = Field(..., description="The web search query, e.g. 'Ollama release notes 2026'")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = ("Search the web for current information. Returns numbered results with "
                        "title, URL and a short snippet. Cite the URL for every fact you use.")
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        return run_search(query)
