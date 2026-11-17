from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_mean_and_model_version():
    response = client.post("/predict", json={"features": [2, 4, 6]})
    assert response.status_code == 200
    assert response.json() == {"prediction": 4.0, "model_version": "demo-v1"}


def test_predict_rejects_empty_features():
    response = client.post("/predict", json={"features": []})
    assert response.status_code == 422
