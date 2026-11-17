from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Cynaris W12D2 ML Prediction API",
    version="1.0.0",
    description="Containerised demonstration ML prediction API",
)

MODEL_VERSION = "demo-v2"


class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "healthy", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.features:
        raise HTTPException(status_code=400, detail="At least one feature is required")
    # Deterministic demo prediction; replace with a loaded, versioned model in production.
    prediction = sum(request.features) / len(request.features)
    return {"prediction": prediction, "model_version": MODEL_VERSION}
