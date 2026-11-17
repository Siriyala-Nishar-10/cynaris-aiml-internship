# W12D2 — Containerising ML Apps: Docker for AI

## What this project builds
A small FastAPI prediction service packaged as a Docker image, with automated
linting, tests, and image-build validation in GitHub Actions.

## Run locally
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
ruff check app tests
python -m pytest -q
```

## Build and run with Docker
```powershell
docker build -t cynaris-w12d2:local .
docker run --rm -p 8000:8000 cynaris-w12d2:local
```

In another PowerShell window:
```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body '{"features":[2,4,6]}'
```

The demo prediction is the arithmetic mean of the supplied features. It is
deliberately deterministic and is not a trained production model.

## CI/CD
`.github/workflows/ci-cd.yml` runs lint, tests, and Docker image build on pushes
and pull requests. Registry publishing is intentionally not enabled until a
registry and repository secrets are configured.

## Monitoring
See [MONITORING.md](MONITORING.md) for metrics, alert conditions, and retraining
triggers.
