from pathlib import Path
from threading import Lock
from time import perf_counter

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"
app = FastAPI(title="W12D5 Production ML API", version="1.0.0")

_model = None
_lock = Lock()
_prediction_count = 0
_error_count = 0
_total_latency_ms = 0.0


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=3, max_length=3)


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                if not MODEL_PATH.exists():
                    raise RuntimeError("Model artifact missing. Run `python train.py` first.")
                _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    global _prediction_count, _error_count, _total_latency_ms
    started = perf_counter()
    try:
        model = get_model()
        values = np.asarray(request.features, dtype=float).reshape(1, -1)
        result = float(model.predict(values)[0])
        _prediction_count += 1
        _total_latency_ms += (perf_counter() - started) * 1000
        return PredictionResponse(prediction=result, model_version="w12d5-v1")
    except Exception as exc:
        _error_count += 1
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metrics")
def metrics():
    average = _total_latency_ms / _prediction_count if _prediction_count else 0.0
    return {
        "prediction_count": _prediction_count,
        "error_count": _error_count,
        "average_latency_ms": round(average, 3),
        "model_version": "w12d5-v1",
    }
