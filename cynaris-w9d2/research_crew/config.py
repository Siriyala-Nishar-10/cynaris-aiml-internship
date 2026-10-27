"""Settings for the research crew. Everything tunable lives here, not in the logic."""

from dataclasses import dataclass

DEFAULT_TOPIC = "Retrieval-Augmented Generation (RAG) best practices for small local LLMs"


@dataclass(frozen=True)
class CrewConfig:
    # CrewAI talks to Ollama through the "ollama/<model>" naming scheme.
    model: str = "ollama/llama3.2:3b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_search_results: int = 5
    # Cap on reasoning/tool-use loops per agent. Small local models can get stuck
    # retrying a tool call, so a low cap fails fast instead of spinning.
    max_iter: int = 6
    verbose: bool = True
