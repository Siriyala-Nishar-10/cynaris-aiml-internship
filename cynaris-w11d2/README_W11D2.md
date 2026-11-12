# W11D2 — MLflow Model Registry & Versioning

## Overview

This project demonstrates MLflow experiment tracking, model registration, version management, and model deployment using a Scikit-learn Random Forest classifier.

**Technology Stack**

- Python
- Scikit-learn
- MLflow 3.18.0
- Pandas
- NumPy

## Learning Objectives

1. Track machine learning experiments using MLflow.
2. Log model parameters, metrics, and artifacts.
3. Compare five experiments with different hyperparameters.
4. Register the best-performing model.
5. Manage model versions using MLflow Model Registry.
6. Assign and use the `champion` model alias.
7. Load and serve a registered model.

## Project Structure

```text
cynaris-w11d2/
│
├── results/
│   ├── experiment_results.csv
│   └── best_model_version.json
│
├── screenshots/
│   ├── 01_experiments.png
│   ├── 02_model_registry.png
│   ├── 03_training_output.png
│   ├── 04_model_serving.png
│   └── 05_api_prediction.png
│
├── main.py
├── requirements.txt
├── README_W11D2.md
└── .gitignore
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Experiment Tracking

The project uses a reproducible classification dataset generated using Scikit-learn.

Five Random Forest experiments were executed with different hyperparameters.

| Experiment | n_estimators | max_depth | Accuracy | F1 Score |
| ---------- | -----------: | --------: | -------: | -------: |
| 1          |           50 |         3 |   0.7967 |   0.8051 |
| 2          |          100 |         5 |   0.8767 |   0.8779 |
| 3          |          150 |         7 |   0.8967 |   0.8963 |
| 4          |          200 |        10 |   0.9133 |   0.9116 |
| 5          |          250 |      None |   0.9133 |   0.9110 |

**Best Experiment:** Experiment 4

**Best F1 Score:** 0.9116

## MLflow Model Registry

The best-performing model was registered using MLflow Model Registry.

| Attribute        | Value                      |
| ---------------- | -------------------------- |
| Registered Model | W11D2-Versioned-Classifier |
| Model Version    | 1                          |
| Source Run       | Experiment-4               |
| Alias            | champion                   |
| Best F1 Score    | 0.9116                     |

The `champion` alias identifies the selected model version.

## Loading the Registered Model

The registered model is loaded using its alias:

```python
registered_model_uri = (
    "models:/W11D2-Versioned-Classifier@champion"
)

loaded_model = mlflow.pyfunc.load_model(
    registered_model_uri
)
```

## Model Serving

The registered model was served using:

```powershell
mlflow models serve `
  -m "models:/W11D2-Versioned-Classifier@champion" `
  --env-manager local `
  --port 5002
```

The prediction API is available at:

```text
http://127.0.0.1:5002/invocations
```

## API Testing

A sample input containing 12 features was submitted to the prediction endpoint.

**Observed prediction:**

```text
predictions
-----------
{1}
```

The API successfully returned a prediction from the registered model.

## Results

- Successfully completed five MLflow experiments.
- Logged parameters, metrics, and model artifacts.
- Compared experiment performance.
- Registered the best-performing model.
- Created model Version 1.
- Assigned the `champion` alias.
- Loaded the registered model.
- Successfully served the model through MLflow.
- Verified the prediction API.

## Screenshots

Evidence screenshots are available in the `screenshots/` directory.

1. MLflow experiment comparison.
2. Registered model and version details.
3. Experiment execution output.
4. Model serving terminal.
5. Successful API prediction.
