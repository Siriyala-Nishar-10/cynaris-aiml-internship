# W11D3 — MLflow Model Serving & REST API

## Setup (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run five experiments and register the best model

```powershell
python main.py
```

This creates `results/experiment_results.csv` and `results/best_model_version.json`.

## MLflow UI

In another activated terminal:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open http://127.0.0.1:5001.

## Start model serving

In another terminal:

```powershell
mlflow models serve -m "models:/W11D3-Serving-Classifier@champion" --env-manager local --port 5003
```

Wait for the server-ready message.

## Test REST API

In a separate PowerShell terminal:

```powershell
.\api_test.ps1
```

Expected response contains a `predictions` field. The exact prediction depends on the model and input.

## Evidence screenshots

Save these in `screenshots/`:

1. `01_experiment_runs.png`
2. `02_registered_model.png`
3. `03_training_output.png`
4. `04_model_server.png`
5. `05_rest_api_test.png`
