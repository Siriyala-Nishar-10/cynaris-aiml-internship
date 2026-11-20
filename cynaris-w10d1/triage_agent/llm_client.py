"""Minimal Ollama chat function for the optional LLM classifier."""
from __future__ import annotations

import requests


def normalise_host(host: str) -> str:
    """OLLAMA_HOST is often written '127.0.0.1:11434' with no scheme."""
    return host if "://" in host else f"http://{host}"


def ollama_chat_fn(model: str, base_url: str, timeout: int = 60, session=None):
    url = f"{normalise_host(base_url).rstrip('/')}/api/chat"
    http = session or requests.Session()

    def chat(prompt: str) -> str:
        try:
            response = http.post(url, json={
                "model": model, "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0},  # classification should be repeatable
            }, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            raise RuntimeError("Could not reach Ollama. Start it with `ollama serve` "
                               f"(or open the Ollama app). URL: {url}") from exc
        if response.status_code == 404:
            raise RuntimeError(f"Model '{model}' not found. Run: ollama pull {model}")
        if response.status_code != 200:
            raise RuntimeError(f"Ollama returned HTTP {response.status_code}")
        return response.json()["message"]["content"]

    return chat
