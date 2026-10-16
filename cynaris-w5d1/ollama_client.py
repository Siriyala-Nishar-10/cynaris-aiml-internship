"""
W5D1: Thin client for Ollama's local REST API
------------------------------------------------
Ollama runs a local server (default http://localhost:11434). This module wraps
the two endpoints we need:

  GET  /api/tags  -> which models are installed locally
  POST /api/chat  -> run inference (non-streaming)

Why raw REST instead of the `ollama` pip package? It shows what actually goes
over the wire (messages, options, timing fields), which is what the viva asks
about. The official package is a thin wrapper around the same endpoints.
"""

import time
from dataclasses import dataclass
from typing import Optional

import requests

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaError(Exception):
    """Base class for all errors raised by this client."""


class OllamaConnectionError(OllamaError):
    """The Ollama server isn't reachable (not installed, not running, wrong port)."""


class OllamaModelNotFoundError(OllamaError):
    """The requested model hasn't been pulled locally."""


@dataclass
class ChatResult:
    model: str
    content: str
    latency_s: float         # wall-clock time for the whole request (includes model load on a cold start)
    load_s: float            # time Ollama spent loading the model into memory (0 when already warm)
    prompt_tokens: int
    completion_tokens: int
    tokens_per_sec: float    # generation speed, from Ollama's own eval timing


def list_models(base_url: str = DEFAULT_BASE_URL, timeout: float = 10) -> list:
    """Return the names of locally installed models, e.g. ['llama3.2:3b']."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaConnectionError(
            f"Could not reach Ollama at {base_url}. Is it running? "
            "Open the Ollama app (or run `ollama serve`) and try again."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise OllamaError(f"Failed to list models: {exc}") from exc
    return [m["name"] for m in resp.json().get("models", [])]


def chat(
    model: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    options: Optional[dict] = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 300,
) -> ChatResult:
    """Send one prompt to a local model and return the reply plus timing stats.

    `system_prompt` sets the model's persona/rules for the conversation.
    `options` are generation settings, e.g. {"temperature": 0.2, "seed": 42,
    "num_predict": 300}. Ollama's per-request timing fields are reported in
    nanoseconds, so they are converted to seconds here.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {"model": model, "messages": messages, "stream": False}
    if options:
        payload["options"] = options

    start = time.perf_counter()
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise OllamaConnectionError(
            f"Could not reach Ollama at {base_url}. Is it running? "
            "Open the Ollama app (or run `ollama serve`) and try again."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError(
            f"Request to {model} timed out after {timeout}s. "
            "Small models on CPU can be slow -- try a shorter prompt or a larger timeout."
        ) from exc
    latency = time.perf_counter() - start

    if resp.status_code == 404:
        raise OllamaModelNotFoundError(
            f"Model '{model}' not found locally. Pull it first: ollama pull {model}"
        )
    if not resp.ok:
        raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    tokens_per_sec = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns else 0.0

    return ChatResult(
        model=model,
        content=data["message"]["content"].strip(),
        latency_s=latency,
        load_s=data.get("load_duration", 0) / 1e9,
        prompt_tokens=data.get("prompt_eval_count", 0),
        completion_tokens=eval_count,
        tokens_per_sec=tokens_per_sec,
    )
