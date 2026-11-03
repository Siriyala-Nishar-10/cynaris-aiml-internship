"""Settings read once from environment variables, with safe defaults."""
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    llm_model: str
    ollama_host: str
    outputs_dir: Path
    max_results: int
    max_iter: int


def _int(env, name, default, minimum):
    raw = env.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def load_settings(env=None) -> Settings:
    """Build Settings from `env` (defaults to os.environ)."""
    env = os.environ if env is None else env
    outputs = Path(env.get("OUTPUTS_DIR", "outputs"))
    return Settings(
        llm_model=env.get("LLM_MODEL", "llama3.2"),
        ollama_host=env.get("OLLAMA_HOST", "http://localhost:11434"),
        outputs_dir=outputs if outputs.is_absolute() else PROJECT_ROOT / outputs,
        max_results=_int(env, "MAX_RESULTS", 5, 1),
        max_iter=_int(env, "MAX_ITER", 6, 1),     # caps how long one agent may loop on a task
    )
