# W12D4 — End-to-End MLOps Pipeline Review

A small FastAPI prediction service packaged with Docker and validated by GitHub Actions.

## Endpoints

- `GET /health` — container/API health
- `POST /predict` — deterministic demo prediction (mean of supplied features)
- `GET /metrics` — in-process request and latency summary

The model is deliberately a deterministic placeholder (`demo-v1`) so the pipeline can be tested without credentials or a model download. Replace the prediction function with a versioned MLflow model for a real deployment.

## Local setup (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
ruff check app tests
python -m pytest -q
uvicorn app.main:app --reload
```

In another PowerShell window:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType "application/json" -Body '{"features":[2,4,6]}'
Invoke-RestMethod http://localhost:8000/metrics
```

## Docker

```powershell
docker build -t cynaris-w12d4:local .
docker run --rm -d --name cynaris-w12d4 -p 8000:8000 cynaris-w12d4:local
docker ps
Invoke-RestMethod http://localhost:8000/health
docker stop cynaris-w12d4
```

## CI/CD

`.github/workflows/ci-cd.yml` runs Ruff and pytest for pushes and pull requests. It builds a Docker image on pushes and pushes `:latest` to GHCR only on `main`. The workflow uses the repository's built-in `GITHUB_TOKEN`; package permissions are declared in the workflow.

For private repositories or organization policy restrictions, confirm that GitHub Actions is allowed to publish packages. Do not commit registry credentials.

## Monitoring

See [MONITORING.md](MONITORING.md). Thresholds are starter values and should be calibrated against real service baselines.
