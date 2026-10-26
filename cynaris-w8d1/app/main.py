"""FastAPI service around the RAG backend: /health, /stats, /query."""
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import backend_name, make_backend

logger = logging.getLogger("rag_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_s: float


def create_app(backend=None) -> FastAPI:
    """Build the app. The backend is created on first use unless one is passed in (tests)."""
    app = FastAPI(title="RAG API", version="1.0.0")
    state = {"backend": backend, "requests": 0, "errors": 0, "total_latency_s": 0.0}

    def get_backend():
        if state["backend"] is None:          # lazy: indexing is slow, so not at import time
            state["backend"] = make_backend()
        return state["backend"]

    @app.get("/health")
    def health():
        name = backend_name() if backend is None else type(backend).__name__
        return {"status": "ok", "backend": name}

    @app.get("/stats")
    def stats():
        n = state["requests"]
        return {"requests": n, "errors": state["errors"],
                "mean_latency_s": round(state["total_latency_s"] / n, 3) if n else 0.0}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        start = time.perf_counter()
        state["requests"] += 1
        try:
            result = get_backend().query(req.question)
        except Exception as exc:
            state["errors"] += 1
            logger.error("query_failed error=%s question_chars=%d",
                         type(exc).__name__, len(req.question))
            raise HTTPException(status_code=500, detail="RAG backend failed") from exc
        latency = time.perf_counter() - start
        state["total_latency_s"] += latency
        logger.info("query_ok latency_s=%.2f sources=%d answer_chars=%d",
                    latency, len(result.sources), len(result.answer))
        return QueryResponse(answer=result.answer, sources=result.sources,
                             latency_s=round(latency, 3))

    return app


app = create_app()
