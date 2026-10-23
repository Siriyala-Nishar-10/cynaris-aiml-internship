"""Retrieval layer: chunking, Ollama embeddings and a persistent vector store."""
from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import Settings


@dataclass
class Chunk:
    source: str
    text: str
    vector: list[float]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows, dropping blank ones."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    pieces = (text[i:i + size] for i in range(0, len(text), step))
    return [p for p in pieces if p.strip()]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def embed(text: str, cfg: Settings) -> list[float]:
    """Embed text with the local Ollama server."""
    req = urllib.request.Request(
        f"{cfg.ollama_url}/api/embeddings",
        data=json.dumps({"model": cfg.embed_model, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)["embedding"]


class VectorMemory:
    """JSON-backed store with brute-force cosine search (fine for a prototype)."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.chunks: list[Chunk] = []
        if self.path.exists():
            self.chunks = [Chunk(**c) for c in json.loads(self.path.read_text(encoding="utf-8"))]

    def add(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)

    def save(self) -> None:
        self.path.write_text(json.dumps([asdict(c) for c in self.chunks]), encoding="utf-8")

    def search(self, query_vec: list[float], k: int) -> list[Chunk]:
        ranked = sorted(self.chunks, key=lambda c: cosine(query_vec, c.vector), reverse=True)
        return ranked[:k]


def ingest_text(name: str, text: str, memory: VectorMemory, cfg: Settings,
                embed_fn=embed) -> int:
    """Chunk and embed one document into memory (not saved); returns chunk count."""
    pieces = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
    for piece in pieces:
        memory.add(Chunk(name, piece, embed_fn(piece, cfg)))
    return len(pieces)


def ingest_folder(folder: Path, memory: VectorMemory, cfg: Settings) -> int:
    """Embed every .txt/.md file under `folder`; returns the chunk count."""
    count = 0
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            count += ingest_text(str(path), path.read_text(encoding="utf-8", errors="ignore"), memory, cfg)
    memory.save()
    return count
