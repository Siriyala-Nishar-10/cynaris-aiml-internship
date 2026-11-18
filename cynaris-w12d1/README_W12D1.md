# W12D1 — MLOps Overview: CI/CD for ML Systems

A container-ready FastAPI example with tests, Docker health checks, GitHub Actions CI/CD, and a monitoring/retraining plan.

## API

- `GET /health` — liveness check
- `POST /predict` — accepts `{"features": [1, 2, 3]}` and returns a demonstration result
- `/docs` — interactive API documentation

**Important:** `/predict` currently calculates the mean of the input features. This is a placeholder to demonstrate API packaging and CI/CD, not a trained ML model.

## Run locally

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
ruff check app tests
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Build and run with Docker

```powershell
docker build -t cynaris-w12d1:local .
docker run --rm -p 8000:8000 cynaris-w12d1:local
```

In another terminal:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body '{"features":[2,4,6]}'
```

## CI/CD

`.github/workflows/ci-cd.yml` runs Ruff linting and pytest for pushes and pull requests. On branch pushes (not PR events), it builds and pushes `ghcr.io/<owner>/cynaris-w12d1:latest` using the GitHub Actions token.

The workflow requires GitHub Actions to have package write permission. Check repository Actions settings if publishing is denied.

## Monitoring

See [MONITORING.md](MONITORING.md) for monitoring signals, example alerts, retraining triggers and safeguards.

## Before production

Replace the demonstration prediction function with a versioned model-loading/inference path, add input validation based on the model schema, and connect real metrics, logs, drift evaluation and deployment approval.
