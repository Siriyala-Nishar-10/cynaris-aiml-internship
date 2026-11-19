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
    assert response.json() == {"prediction": 4.0, "model_version": "demo-v2"}


def test_prediction_requires_features():
    response = client.post("/predict", json={"features": []})
    assert response.status_code == 422
