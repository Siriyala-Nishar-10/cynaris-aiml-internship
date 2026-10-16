"""
W4D3: Model Serialisation — joblib & pickle
Task: train a model, save it with both joblib and pickle, reload it,
verify predictions match, and demonstrate why serialisation matters
for ML deployment.

Author: Siriyala Nishar
"""

from pathlib import Path
import pickle
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_pipeline(csv_path: Path) -> tuple:
    """Load data, train a full sklearn Pipeline, and return test data."""

    df = pd.read_csv(csv_path)

    df_encoded = pd.get_dummies(
        df,
        columns=["employment_type"],
        drop_first=True
    )

    feature_cols = [
        column for column in df_encoded.columns
        if column != "loan_approved"
    ]

    X = df_encoded[feature_cols]
    y = df_encoded["loan_approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    print("=== Trained Pipeline Performance ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, probabilities):.3f}")

    return pipeline, X_test, y_test


def save_with_joblib(pipeline, path: Path) -> None:
    """Save the trained pipeline using joblib."""

    start = time.time()

    joblib.dump(pipeline, path)

    elapsed = time.time() - start
    size_kb = path.stat().st_size / 1024

    print(
        f"\njoblib save: {elapsed * 1000:.1f} ms, "
        f"{size_kb:.1f} KB → {path}"
    )


def save_with_pickle(pipeline, path: Path) -> None:
    """Save the trained pipeline using Python pickle."""

    start = time.time()

    with open(path, "wb") as file:
        pickle.dump(pipeline, file)

    elapsed = time.time() - start
    size_kb = path.stat().st_size / 1024

    print(
        f"pickle save: {elapsed * 1000:.1f} ms, "
        f"{size_kb:.1f} KB → {path}"
    )


def load_and_verify(
    joblib_path: Path,
    pickle_path: Path,
    X_test,
    y_test
) -> None:
    """Reload both models and verify their predictions."""

    pipeline_joblib = joblib.load(joblib_path)

    with open(pickle_path, "rb") as file:
        pipeline_pickle = pickle.load(file)

    predictions_joblib = pipeline_joblib.predict(X_test)
    predictions_pickle = pipeline_pickle.predict(X_test)

    print("\n=== Reload Verification ===")
    print(
        f"joblib accuracy: "
        f"{accuracy_score(y_test, predictions_joblib):.3f}"
    )
    print(
        f"pickle accuracy: "
        f"{accuracy_score(y_test, predictions_pickle):.3f}"
    )

    identical = np.array_equal(
        predictions_joblib,
        predictions_pickle
    )

    print(f"Predictions identical: {identical}")

    if identical:
        print("Both reloaded models produce the same predictions ✓")


def simulate_inference(joblib_path: Path, X_test) -> None:
    """Simulate production inference using the saved model."""

    pipeline = joblib.load(joblib_path)

    sample = X_test.iloc[:3]

    predictions = pipeline.predict(sample)
    probabilities = pipeline.predict_proba(sample)[:, 1]

    print("\n=== Simulated Inference (3 new applicants) ===")

    for i, (prediction, probability) in enumerate(
        zip(predictions, probabilities)
    ):
        decision = "APPROVED" if prediction == 1 else "REJECTED"

        print(
            f"Applicant {i + 1}: "
            f"{decision} (confidence: {probability:.1%})"
        )


def main() -> None:
    csv_path = Path("loan_evaluation.csv")

    joblib_path = Path("loan_model.joblib")
    pickle_path = Path("loan_model.pkl")

    pipeline, X_test, y_test = train_pipeline(csv_path)

    save_with_joblib(pipeline, joblib_path)
    save_with_pickle(pipeline, pickle_path)

    load_and_verify(
        joblib_path,
        pickle_path,
        X_test,
        y_test
    )

    simulate_inference(joblib_path, X_test)

    print("\n=== Summary ===")
    print("Trained pipeline saved in 2 formats:")
    print(f"  {joblib_path} — recommended for sklearn models")
    print(f"  {pickle_path} — Python standard serialization")
    print("Both reload correctly and produce identical predictions.")


if __name__ == "__main__":
    main()