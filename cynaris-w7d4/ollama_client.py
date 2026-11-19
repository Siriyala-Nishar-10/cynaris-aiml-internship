"""Tiny helper around the Ollama REST API (http://localhost:11434)."""
import os, time
import requests

HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def installed_models():
    r = requests.get(f"{HOST}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def chat(model, user, system=None, temperature=0.2, max_tokens=300):
    """One non-streaming chat call. Returns text plus timing/token stats from Ollama."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": temperature, "num_predict": max_tokens}}
    t0 = time.perf_counter()
    r = requests.post(f"{HOST}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    d = r.json()
    wall = time.perf_counter() - t0
    out_tokens = d.get("eval_count", 0)
    gen_s = d.get("eval_duration", 0) / 1e9
    return {"text": d["message"]["content"].strip(), "wall_s": round(wall, 2),
            "load_s": round(d.get("load_duration", 0) / 1e9, 2),
            "out_tokens": out_tokens,
            "tokens_per_s": round(out_tokens / gen_s, 1) if gen_s else 0.0}
