from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Cynaris W12D1 ML API",
    description="Container-ready example API for the MLOps CI/CD capstone.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    # Replace these example features with the inputs required by your trained model.
    features: list[float] = Field(min_length=1, description="Numeric model features")


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness/health endpoint used by Docker and deployment checks."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Demonstration prediction: returns the mean of the supplied features.

    This is deliberately a placeholder, not a trained ML model. Replace this
    function with loading and invoking your versioned model before production.
    """
    result = sum(request.features) / len(request.features)
    return PredictionResponse(prediction=result, model_version="demo-v1")
