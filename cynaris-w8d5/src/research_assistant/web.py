"""Local web UI: a tiny stdlib HTTP server around the same pipeline.

    python -m research_assistant.web      ->  http://127.0.0.1:8000

Binds to localhost only; nothing leaves your machine except calls to Ollama.
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import Settings
from .memory import VectorMemory, ingest_folder, ingest_text
from .service import answer_question

INDEX = Path(__file__).parent / "static" / "index.html"
ALLOWED_SUFFIXES = {".txt", ".md"}


class App:
    """Shared state for all requests: settings, memory and a write lock."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.memory = VectorMemory(cfg.store_path)
        self.lock = threading.Lock()

    def status(self) -> dict:
        files = sorted({c.source for c in self.memory.chunks})
        return {"chunks": len(self.memory.chunks), "files": files,
                "chat_model": self.cfg.chat_model, "embed_model": self.cfg.embed_model}

    def ask(self, body: dict) -> dict:
        question = str(body.get("question", "")).strip()
        if not question:
            raise ValueError("Please enter a question.")
        return answer_question(question, self.cfg, self.memory,
                               evaluate=bool(body.get("evaluate")))

    def upload(self, body: dict) -> dict:
        added = 0
        with self.lock:
            for f in body.get("files", []):
                name = os.path.basename(str(f.get("name", "")))
                if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
                    raise ValueError(f"{name or 'file'}: only .txt and .md files are supported.")
                added += ingest_text(name, str(f.get("text", "")), self.memory, self.cfg)
            self.memory.save()
        return {"added": added, **self.status()}

    def ingest_path(self, body: dict) -> dict:
        folder = Path(str(body.get("folder", ""))).expanduser()
        if not folder.is_dir():
            raise ValueError(f"Folder not found: {folder}")
        with self.lock:
            added = ingest_folder(folder, self.memory, self.cfg)
        return {"added": added, **self.status()}

    def clear(self) -> dict:
        with self.lock:
            self.memory.chunks.clear()
            self.memory.save()
        return self.status()


def make_handler(app: App):
    routes = {"/api/ask": app.ask, "/api/upload": app.upload,
              "/api/ingest": app.ingest_path, "/api/clear": lambda _: app.clear()}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self._json(200, app.status())
            else:
                self._json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            handler = routes.get(self.path)
            if not handler:
                return self._json(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                self._json(200, handler(body))
            except ValueError as exc:  # bad input (includes JSON errors)
                self._json(400, {"error": str(exc)})
            except OSError as exc:  # Ollama unreachable / timeout
                self._json(502, {"error": f"Could not reach Ollama at {app.cfg.ollama_url}: {exc}"})
            except Exception as exc:  # keep the server alive on pipeline errors
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt, *args) -> None:  # quiet console
            pass

    return Handler


def main() -> None:
    host, port = "127.0.0.1", int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), make_handler(App(Settings())))
    print(f"Research assistant running at http://{host}:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
