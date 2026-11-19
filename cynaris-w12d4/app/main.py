from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("cynaris.ml_api")

app = FastAPI(title="Cynaris MLOps API", version="1.0.0")
_recent_latencies: deque[float] = deque(maxlen=500)
_prediction_count = 0


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=1)


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    latency_ms: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    values = list(_recent_latencies)
    return {
        "prediction_count": _prediction_count,
        "latency_samples": len(values),
        "average_latency_ms": round(sum(values) / len(values), 3) if values else 0.0,
        "p95_latency_ms": round(sorted(values)[int(0.95 * (len(values) - 1))], 3)
        if values else 0.0,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    global _prediction_count
    started = time.perf_counter()
    try:
        # Deterministic demo model. Replace with a versioned MLflow model in production.
        prediction = sum(payload.features) / len(payload.features)
        latency_ms = (time.perf_counter() - started) * 1000
        _prediction_count += 1
        _recent_latencies.append(latency_ms)
        logger.info(
            "prediction_completed",
            extra={
                "model_version": os.getenv("MODEL_VERSION", "demo-v1"),
                "latency_ms": round(latency_ms, 3),
                "feature_count": len(payload.features),
            },
        )
        return PredictionResponse(
            prediction=round(prediction, 6),
            model_version=os.getenv("MODEL_VERSION", "demo-v1"),
            latency_ms=round(latency_ms, 3),
        )
    except (TypeError, ValueError) as exc:
        logger.exception("prediction_failed")
        raise HTTPException(status_code=400, detail="Invalid prediction input") from exc
