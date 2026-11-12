"""FastAPI service for Haystack search: /health, /info, /search."""
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.search import ModeUnavailable, SearchService

logger = logging.getLogger("haystack_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Hit(BaseModel):
    text: str
    source: str
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[Hit]
    latency_s: float


def create_app(service=None) -> FastAPI:
    """Build the app. The index is built at start-up unless a service is passed in (tests)."""
    state = {"service": service}

    def get_service():
        if state["service"] is None:
            state["service"] = SearchService()
        return state["service"]

    @asynccontextmanager
    async def lifespan(app):
        get_service()                      # build the index once, before serving traffic
        yield

    app = FastAPI(title="Haystack Search API", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/info")
    def info():
        svc = get_service()
        return {"documents": svc.n_documents, "chunks": svc.n_chunks, "modes": svc.modes()}

    @app.get("/search", response_model=SearchResponse)
    def search(q: str = Query(min_length=2, max_length=300),
               mode: Literal["bm25", "dense"] = "bm25",
               top_k: int = Query(default=3, ge=1, le=10)):
        start = time.perf_counter()
        try:
            results = get_service().search(q, mode=mode, top_k=top_k)
        except ModeUnavailable as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        latency = round(time.perf_counter() - start, 4)
        logger.info("search mode=%s top_k=%d hits=%d latency_s=%.3f",
                    mode, top_k, len(results), latency)
        return SearchResponse(query=q, mode=mode, results=[Hit(**r) for r in results],
                              latency_s=latency)

    return app


app = create_app()
