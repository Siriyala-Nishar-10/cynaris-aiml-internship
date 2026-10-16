"""Pydantic schemas for the model-serving API."""

from typing import List

from pydantic import BaseModel, Field, field_validator

N_FEATURES = 10


class PredictRequest(BaseModel):
    """A single row of feature values, in the exact order the model expects."""

    features: List[float] = Field(
        ...,
        description=f"Exactly {N_FEATURES} numeric feature values, in training order.",
        examples=[[0.5, -1.2, 0.3, 2.1, -0.4, 0.0, 1.1, -0.9, 0.2, 0.7]],
    )

    @field_validator("features")
    @classmethod
    def check_length(cls, v: List[float]) -> List[float]:
        if len(v) != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} features, got {len(v)}")
        return v


class BatchPredictRequest(BaseModel):
    """Multiple rows in a single request."""

    instances: List[List[float]] = Field(
        ...,
        description=f"A list of feature rows, each with exactly {N_FEATURES} values.",
    )

    @field_validator("instances")
    @classmethod
    def check_shape(cls, v: List[List[float]]) -> List[List[float]]:
        if not v:
            raise ValueError("instances must contain at least one row")
        for i, row in enumerate(v):
            if len(row) != N_FEATURES:
                raise ValueError(
                    f"Row {i} has {len(row)} features, expected {N_FEATURES}"
                )
        return v


class PredictResponse(BaseModel):
    probability: float = Field(..., description="Predicted probability of the positive class")
    prediction: int = Field(..., description="0 or 1, using the tuned decision threshold")
    threshold_used: float


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str | None = None
    threshold: float | None = None
