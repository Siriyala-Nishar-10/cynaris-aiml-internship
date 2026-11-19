# W12D3 — Monitoring ML Models in Production

A small FastAPI prediction service packaged with Docker, tested with pytest, linted
with Ruff, and built/published by GitHub Actions to GitHub Container Registry (GHCR).

## Project structure

```text
.
├── .github/workflows/ci-cd.yml
├── app/main.py
├── tests/test_api.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-dev.txt
└── MONITORING.md
```

## Run locally (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
ruff check app tests
python -m pytest -q
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for API documentation.

## Test the API (PowerShell)

```powershell
Invoke-RestMethod http://localhost:8000/health

Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body '{"features":[2,4,6]}'

Invoke-RestMethod http://localhost:8000/metrics
```

Expected demo prediction for `[2,4,6]`: `4.0`.

## Docker

Make sure Docker Desktop is running, then:

```powershell
docker build -t cynaris-w12d3:local .
docker run --rm -d --name cynaris-w12d3 -p 8000:8000 cynaris-w12d3:local
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType "application/json" -Body '{"features":[2,4,6]}'
docker logs cynaris-w12d3
docker stop cynaris-w12d3
```

## CI/CD

The workflow runs Ruff and pytest on pushes and pull requests. On branch pushes
(and manual runs), it builds and pushes the image to GHCR. GitHub Actions provides
`GITHUB_TOKEN`; the workflow grants `packages: write`.

After a successful workflow, find the image under the repository's **Packages**
section. If publishing is blocked, check repository Actions permissions and package
visibility.

## Important limitation

The prediction function is a deterministic placeholder (`mean(features)`), not a
trained ML model. Replace it with the approved model artifact and add model-specific
validation before production use. `/metrics` is an in-memory demo endpoint only.

See [MONITORING.md](MONITORING.md) for the monitoring, alerting, and retraining plan.
