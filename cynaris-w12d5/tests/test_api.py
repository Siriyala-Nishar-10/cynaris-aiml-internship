from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint():
    # CI and Docker build run train.py before tests.
    response = client.post("/predict", json={"features": [2.0, 4.0, 6.0]})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["prediction"], float)
    assert body["model_version"] == "w12d5-v1"

def test_invalid_feature_count():
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "prediction_count" in response.json()
