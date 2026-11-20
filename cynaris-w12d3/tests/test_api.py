from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    response = client.post("/predict", json={"features": [2, 4, 6]})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 4.0
    assert body["model_version"] == "demo-v1"
    assert body["latency_ms"] >= 0


def test_metrics_endpoint():
    client.post("/predict", json={"features": [1, 3]})
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["prediction_requests_total"] >= 1
    assert "latency_ms_p95" in body
