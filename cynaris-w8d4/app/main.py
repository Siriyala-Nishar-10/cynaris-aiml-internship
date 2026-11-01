"""FastAPI service for Haystack search: /health, /info, /search."""
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.search import ModeUnavailable, SearchService

logger = logging.getLogger("haystack_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MIN_QUERY_CHARS = 2


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
    lock = threading.Lock()

    def get_service():
        if state["service"] is None:
            with lock:                          # two first requests must not build two indexes
                if state["service"] is None:
                    state["service"] = SearchService()
        return state["service"]

    @asynccontextmanager
    async def lifespan(app):
        get_service()                           # build the index once, before serving traffic
        yield

    app = FastAPI(title="Haystack Search API", version="1.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/info")
    def info():
        svc = get_service()
        return {"documents": svc.n_documents, "chunks": svc.n_chunks, "modes": svc.modes()}

    @app.get("/search", response_model=SearchResponse)
    def search(q: str = Query(min_length=MIN_QUERY_CHARS, max_length=300),
               mode: Literal["bm25", "dense"] = "bm25",
               top_k: int = Query(default=3, ge=1, le=10)):
        query = q.strip()
        if len(query) < MIN_QUERY_CHARS:         # "  " passes the length check above
            raise HTTPException(status_code=422,
                                detail="query must contain at least 2 non-space characters")
        start = time.perf_counter()
        try:
            results = get_service().search(query, mode=mode, top_k=top_k)
        except ModeUnavailable as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("search_failed mode=%s top_k=%d query_chars=%d",
                             mode, top_k, len(query))
            raise HTTPException(status_code=500, detail="search failed") from exc
        latency = round(time.perf_counter() - start, 4)
        logger.info("search mode=%s top_k=%d hits=%d latency_s=%.3f",
                    mode, top_k, len(results), latency)
        return SearchResponse(query=query, mode=mode, results=[Hit(**r) for r in results],
                              latency_s=latency)

    return app


app = create_app()
