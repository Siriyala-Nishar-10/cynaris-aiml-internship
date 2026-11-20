# W12D5 — 3M Capstone: Production AI System — Full Pipeline

A small production-style ML API demonstrating model training and tracking with MLflow, FastAPI inference, Docker packaging, CI validation, and container publishing.

## Architecture

1. `train.py` creates a deterministic demonstration dataset, trains a scikit-learn LinearRegression model, logs parameters/metrics/model to local MLflow tracking, and saves `artifacts/model.joblib`.
2. FastAPI loads the saved artifact and exposes `/health`, `/predict`, and `/metrics`.
3. Docker builds the application and model artifact into a non-root container.
4. GitHub Actions runs Ruff, trains the artifact, executes pytest, builds the image, and publishes to GHCR on pushes to `main` or version tags.

## Run locally (PowerShell)

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements-dev.txt
python train.py
ruff check app tests train.py
python -m pytest -q
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

Example request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"features":[2,4,6]}'
```

## Docker

```powershell
docker build -t cynaris-w12d5:local .
docker run --rm -p 8000:8000 cynaris-w12d5:local
```

Then visit `http://localhost:8000/health` and `http://localhost:8000/docs`.

## MLflow UI

After `python train.py`, run:

```powershell
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`.

## CI/CD publishing

Pull requests run lint, training, tests and Docker build. Pushes to `main` and version tags also publish `ghcr.io/<owner>/cynaris-w12d5:latest`. Ensure the repository allows GitHub Actions to write packages.
