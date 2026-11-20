"""Central configuration. Every knob is an environment variable (MLOps: no
hard-coded settings, so the same code runs locally, in CI and in prod)."""
import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", "llama3.2"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "nomic-embed-text"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "100")))
    store_path: str = field(default_factory=lambda: _env("STORE_PATH", "memory.json"))
    mlflow_uri: str = field(default_factory=lambda: _env("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    experiment: str = field(default_factory=lambda: _env("MLFLOW_EXPERIMENT", "research-assistant"))



# Keep the pipeline fully local: switch off library telemetry.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")        # CrewAI
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")       # Ragas
