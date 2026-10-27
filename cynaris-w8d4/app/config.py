"""Runtime configuration, read once from environment variables and validated."""
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_flag(value, default=False):
    """Turn an environment value such as '1', 'true' or 'off' into a bool."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(env, name, default):
    raw = env.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    docs_dir: Path
    enable_dense: bool
    embed_model: str
    chunk_words: int
    chunk_overlap: int


def load_settings(env=None) -> Settings:
    """Build Settings from `env` (defaults to os.environ).

    A relative DOCS_DIR is resolved against the project root, not the current
    working directory, so the app behaves the same however it is started.
    """
    env = os.environ if env is None else env
    docs_dir = Path(env.get("DOCS_DIR", "sample_docs"))
    if not docs_dir.is_absolute():
        docs_dir = PROJECT_ROOT / docs_dir
    chunk_words = _positive_int(env, "CHUNK_WORDS", 150)
    chunk_overlap = _positive_int(env, "CHUNK_OVERLAP", 20)
    if chunk_words < 1:
        raise ValueError("CHUNK_WORDS must be at least 1")
    if chunk_overlap >= chunk_words:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_WORDS")
    return Settings(
        docs_dir=docs_dir,
        enable_dense=parse_flag(env.get("ENABLE_DENSE")),
        embed_model=env.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        chunk_words=chunk_words,
        chunk_overlap=chunk_overlap,
    )
