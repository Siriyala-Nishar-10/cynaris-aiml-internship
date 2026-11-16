# W11D1 — MLflow Experiment Tracking: Setup & Logging

## Objective

Instrument a scikit-learn model with MLflow, run five hyperparameter experiments,
compare metrics, and register the best model in the MLflow Model Registry.

## Stack

- Python
- scikit-learn
- MLflow
- pandas
- pytest

## Project structure

```text
cynaris-w11d1/
├── tests/
│   └── test_project.py
├── results/                 # generated after running main.py
├── .env.example
├── .gitignore
├── main.py
├── README_W11D1.md
└── requirements.txt
```

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run five experiments

```powershell
python main.py
```

The script creates a reproducible synthetic classification dataset and trains
five RandomForestClassifier configurations. Each MLflow run logs:

- Hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`)
- Accuracy, precision, recall, and F1 score
- The trained sklearn model artifact and model signature

The best run is selected by F1 score and registered as
`W11D1-Best-Classifier`. Generated comparison output is written to
`results/experiment_results.csv`, and registration details to
`results/best_model.json`.

## Compare runs in MLflow UI

After running the experiments, open a second terminal in the project directory:

```powershell
.\.venv\Scripts\Activate.ps1
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000`, select the `W11D1-Sklearn-Experiments`
experiment, and compare the five runs by their logged parameters and metrics.

## Load the registered model

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/W11D1-Best-Classifier/1")
print(model.predict([[0.0] * 12]))
```

The example input is only a shape demonstration; use 12 numeric features.

## Serve the registered model

With the MLflow UI stopped if port 5000 is already occupied:

```powershell
mlflow models serve -m "models:/W11D1-Best-Classifier/1" -p 5001 --env-manager local
```

The serving process exposes the model locally. For a quick health check, visit
`http://127.0.0.1:5001` or send a prediction request using the MLflow scoring
API. Keep the local service for development only.

## Tests

```powershell
python -m pytest -q
```

## Notes

- The dataset is generated locally, so no external dataset download is needed.
- The SQLite tracking backend is used because the Model Registry requires a
  database-backed store.
- The project logs a local model; it does not deploy to a public server.
