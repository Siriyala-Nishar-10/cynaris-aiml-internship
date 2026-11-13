import json
from pathlib import Path

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


# Configuration
EXPERIMENT_NAME = "W11D2-Model-Versioning"
MODEL_NAME = "W11D2-Versioned-Classifier"

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient()


# Generate dataset
X, y = make_classification(
    n_samples=1200,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    random_state=42,
)

X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

# Five experiment configurations
experiments = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 7},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 250, "max_depth": None},
]

results = []

print("\nW11D2 — MLflow Model Registry & Versioning")
print("=" * 50)

# Run experiments
for index, params in enumerate(experiments, start=1):

    with mlflow.start_run(run_name=f"Experiment-{index}") as run:

        model = RandomForestClassifier(
            **params,
            random_state=42,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        mlflow.log_params(params)

        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

        signature = infer_signature(X_test, predictions)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.head(5),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        results.append({
            "run_id": run.info.run_id,
            "experiment": index,
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

        print(
            f"Experiment {index}: "
            f"Accuracy={accuracy:.4f}, F1={f1:.4f}"
        )


# Compare experiments
results_df = pd.DataFrame(results)
results_df.to_csv(
    RESULTS_DIR / "experiment_results.csv",
    index=False,
)

best_result = max(results, key=lambda item: item["f1_score"])

print("\nBest Experiment:")
print(best_result)


# Register the best model
model_uri = f"runs:/{best_result['run_id']}/model"

model_version = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)

print("\nModel Registration:")
print(f"Model Name: {MODEL_NAME}")
print(f"Model Version: {model_version.version}")


# Assign champion alias to this version
client.set_registered_model_alias(
    MODEL_NAME,
    "champion",
    model_version.version,
)

print("\nChampion alias assigned successfully.")


# Load the registered model using its alias
registered_model_uri = f"models:/{MODEL_NAME}@champion"

loaded_model = mlflow.pyfunc.load_model(registered_model_uri)

sample_predictions = loaded_model.predict(X_test.head(5))

print("\nLoaded Registered Model Predictions:")
print(sample_predictions)


# Save model version information
version_info = {
    "model_name": MODEL_NAME,
    "version": model_version.version,
    "alias": "champion",
    "run_id": best_result["run_id"],
    "best_f1_score": best_result["f1_score"],
    "model_uri": registered_model_uri,
}

with open(RESULTS_DIR / "best_model_version.json", "w") as file:
    json.dump(version_info, file, indent=4)

print("\nAll five experiments completed successfully.")
print("Model registered, versioned, and loaded.")