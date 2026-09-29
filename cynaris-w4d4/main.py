"""
app/main.py

FastAPI service that loads the artifacts produced by train_model.py
(model.joblib, scaler.joblib, metadata.json) and exposes them as a
prediction API.

Run locally:
    uvicorn app.main:app --reload --port 8000

Then visit http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_service")

MODEL_DIR = Path(__file__).resolve().parent / "model"

# Populated at startup by load_artifacts(); kept in a small mutable holder
# (rather than plain globals) so it's easy to reset in tests.
class ModelState:
    model = None
    scaler = None
    metadata: Optional[dict] = None


state = ModelState()


def load_artifacts() -> None:
    try:
        state.model = joblib.load(MODEL_DIR / "model.joblib")
        state.scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        with open(MODEL_DIR / "metadata.json") as f:
            state.metadata = json.load(f)
        logger.info(
            "Loaded model=%s threshold=%.2f",
            state.metadata.get("model_type"),
            state.metadata.get("threshold"),
        )
    except FileNotFoundError as exc:
        logger.error(
            "Model artifacts not found in %s. Run `python train_model.py` first.",
            MODEL_DIR,
        )
        raise exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the model once, keep it in memory for the life of the process.
    load_artifacts()
    yield
    # Shutdown: nothing to clean up, but this is where you'd close DB pools etc.


app = FastAPI(
    title="Imbalanced-Data Model Serving API",
    description=(
        "Serves predictions from a classifier trained with class-weight "
        "balancing and a tuned decision threshold for an imbalanced target."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _predict_one(features: list[float]) -> PredictResponse:
    if state.model is None or state.scaler is None or state.metadata is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    x = np.asarray(features, dtype=float).reshape(1, -1)
    x_scaled = state.scaler.transform(x)
    proba = float(state.model.predict_proba(x_scaled)[0, 1])
    threshold = float(state.metadata["threshold"])
    prediction = int(proba >= threshold)
    return PredictResponse(probability=proba, prediction=prediction, threshold_used=threshold)


@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({"service": "model-serving-api", "docs": "/docs"})


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness/readiness check — used by orchestrators (k8s, load balancers, etc.)."""
    loaded = state.model is not None
    return HealthResponse(
        status="ok" if loaded else "model_not_loaded",
        model_loaded=loaded,
        model_type=state.metadata.get("model_type") if state.metadata else None,
        threshold=state.metadata.get("threshold") if state.metadata else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict for a single row of features."""
    try:
        return _predict_one(request.features)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    """Predict for multiple rows in one call, to avoid N round trips."""
    try:
        results = [_predict_one(row) for row in request.instances]
        return BatchPredictResponse(results=results)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=400, detail=str(exc))
