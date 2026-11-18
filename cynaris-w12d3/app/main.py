from __future__ import annotations

import logging
import os
import time
from collections import deque
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("ml-api")

app = FastAPI(
    title="Cynaris W12D3 Monitored ML API",
    version="1.0.0",
    description="Demo prediction API with lightweight operational monitoring.",
)

# Demo-only in-memory metrics. Replace with Prometheus/OpenTelemetry in production.
_metrics_lock = Lock()
_recent_latencies_ms: deque[float] = deque(maxlen=500)
_request_count = 0
_error_count = 0


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    latency_ms: float


def predict(features: list[float]) -> float:
    """Deterministic demo model; replace with a trained model artifact."""
    return round(sum(features) / len(features), 4)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    with _metrics_lock:
        latencies = sorted(_recent_latencies_ms)
        count = _request_count
        errors = _error_count

    def percentile(values: list[float], pct: float) -> float | None:
        if not values:
            return None
        index = min(int((len(values) - 1) * pct), len(values) - 1)
        return round(values[index], 2)

    return {
        "prediction_requests_total": count,
        "prediction_errors_total": errors,
        "error_rate": round(errors / count, 4) if count else 0.0,
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "model_version": os.getenv("MODEL_VERSION", "demo-v1"),
        "note": "In-memory demo metrics reset when the process restarts.",
    }


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(payload: PredictionRequest) -> PredictionResponse:
    global _request_count, _error_count
    start = time.perf_counter()
    try:
        result = predict(payload.features)
        latency_ms = (time.perf_counter() - start) * 1000
        with _metrics_lock:
            _request_count += 1
            _recent_latencies_ms.append(latency_ms)
        logger.info("prediction_completed model_version=%s latency_ms=%.2f",
                    os.getenv("MODEL_VERSION", "demo-v1"), latency_ms)
        return PredictionResponse(
            prediction=result,
            model_version=os.getenv("MODEL_VERSION", "demo-v1"),
            latency_ms=round(latency_ms, 2),
        )
    except Exception as exc:
        with _metrics_lock:
            _request_count += 1
            _error_count += 1
        logger.exception("prediction_failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
