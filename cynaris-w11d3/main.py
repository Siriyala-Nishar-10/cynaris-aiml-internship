
from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from mlflow.models import infer_signature

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODEL_NAME = "W11D3-Serving-Classifier"
EXPERIMENT_NAME = "W11D3-MLflow-Serving"


def main():

    # 1. Configure MLflow
    mlflow.set_tracking_uri(
        f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Generate dataset
    X, y = make_classification(
        n_samples=1200,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    columns = [
        f"feature_{i}" for i in range(X.shape[1])
    ]

    X_train = pd.DataFrame(
        X_train, columns=columns
    )

    X_test = pd.DataFrame(
        X_test, columns=columns
    )

    # 3. Define five hyperparameter configurations
    configs = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 150, "max_depth": 7},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 250, "max_depth": None},
    ]

    rows = []
    best_run_id = None
    best_f1 = -1.0

    # 4. Run five experiments
    for number, params in enumerate(configs, start=1):

        with mlflow.start_run(
            run_name=f"experiment_{number}"
        ) as run:

            model = RandomForestClassifier(
                **params,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            # 5. Calculate evaluation metrics
            metrics = {
                "accuracy": accuracy_score(
                    y_test, predictions
                ),
                "precision": precision_score(
                    y_test, predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test, predictions,
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    y_test, predictions,
                    zero_division=0,
                ),
            }

            # 6. Log parameters and metrics
            mlflow.log_params({
                key: ("None" if value is None else value)
                for key, value in params.items()
            })

            mlflow.log_param("random_state", 42)

            mlflow.log_metrics(metrics)

            # 7. Log model artifact
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=infer_signature(
                    X_test,
                    model.predict(X_test),
                ),
                input_example=X_test.head(3),
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ],
            )

            rows.append({
                "experiment": number,
                "run_id": run.info.run_id,
                **params,
                **metrics,
            })

            print(
                f"Experiment {number}: "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1_score']:.4f}"
            )

            # 8. Identify best model
            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_run_id = run.info.run_id

    # 9. Save experiment comparison
    results = pd.DataFrame(rows)

    results = results.sort_values(
        "f1_score",
        ascending=False,
    )

    results.to_csv(
        RESULTS / "experiment_results.csv",
        index=False,
    )

    print("\nExperiment comparison saved.")

    # 10. Register best model
    model_uri = f"runs:/{best_run_id}/model"

    print("\nRegistering best model...")

    registered = mlflow.register_model(
        model_uri,
        MODEL_NAME,
    )

    # 11. Assign champion alias
    client = MlflowClient()

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        registered.version,
    )

    # 12. Save model version details
    details = {
        "registered_model_name": MODEL_NAME,
        "version": str(registered.version),
        "alias": "champion",
        "best_run_id": best_run_id,
        "best_f1_score": float(best_f1),
        "model_uri": f"models:/{MODEL_NAME}@champion",
    }

    with open(
        RESULTS / "best_model_version.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(details, file, indent=4)

    # 13. Load registered model
    loaded_model = mlflow.pyfunc.load_model(
        details["model_uri"]
    )

    sample_predictions = loaded_model.predict(
        X_test.head(5)
    )

    print("\n========== FINAL RESULTS ==========")
    print("Model:", MODEL_NAME)
    print("Version:", registered.version)
    print("Alias: champion")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(
        "Sample Predictions:",
        sample_predictions.tolist(),
    )

    print("\nStart model serving with:")

    print(
        f'mlflow models serve -m '
        f'"models:/{MODEL_NAME}@champion" '
        f'--env-manager local --port 5003'
    )


if __name__ == "__main__":
    main()
