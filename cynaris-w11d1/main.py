"""Run five sklearn experiments, compare them, and register the best model."""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "W11D1-Sklearn-Experiments"
REGISTERED_MODEL_NAME = "W11D1-Best-Classifier"
TRACKING_DIR = Path("mlruns")
ARTIFACT_DIR = Path("artifacts")
RESULTS_DIR = Path("results")


def build_dataset():
    """Create a reproducible classification dataset for the exercise."""
    X, y = make_classification(
        n_samples=1200,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        weights=[0.58, 0.42],
        class_sep=1.0,
        random_state=42,
    )
    return train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )


def main():
    TRACKING_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Local SQLite backend supports the MLflow Model Registry.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = build_dataset()
    configs = [
        {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
        {"n_estimators": 150, "max_depth": 7, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 10, "min_samples_split": 4},
        {"n_estimators": 250, "max_depth": None, "min_samples_split": 5},
    ]

    rows = []
    best_f1 = -1.0
    best_run_id = None

    for index, params in enumerate(configs, start=1):
        with mlflow.start_run(run_name=f"random-forest-{index}") as run:
            model = RandomForestClassifier(
                **params, random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
            }
            mlflow.log_params(params)
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_metrics(metrics)

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train[:5],
            skops_trusted_types=["sklearn.tree._tree.Tree"],
         )
            rows.append({
                "run_id": run.info.run_id,
                **params,
                **metrics,
            })
            print(f"Run {index}: {metrics}")

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_run_id = run.info.run_id

    results = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
    results.to_csv(RESULTS_DIR / "experiment_results.csv", index=False)

    # Register the best run's logged model artifact.
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "registered_model_version": registered.version,
        "best_run_id": best_run_id,
        "best_f1_score": best_f1,
        "runs_completed": len(configs),
    }
    (RESULTS_DIR / "best_model.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nExperiment comparison:")
    print(results.to_string(index=False))
    print("\nBest model registered:")
    print(json.dumps(summary, indent=2))
    print("\nStart the MLflow UI with:")
    print("mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
