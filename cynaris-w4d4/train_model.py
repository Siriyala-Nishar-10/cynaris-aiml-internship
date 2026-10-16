"""
train_model.py

Trains a binary classifier on an imbalanced dataset and saves everything
the FastAPI service needs to serve predictions:

    model/model.joblib        - the fitted classifier
    model/scaler.joblib       - the fitted StandardScaler
    model/metadata.json       - feature names, chosen threshold, and metrics

Imbalance handling strategy used here (documented for the viva questions):
  1. class_weight="balanced" on the classifier itself, so the loss function
     penalises mistakes on the minority class more heavily. This is usually
     the first thing to try before resampling, because it doesn't throw away
     or duplicate any data.
  2. Threshold tuning: the default 0.5 cutoff on predict_proba is rarely
     optimal for imbalanced problems. We sweep thresholds and pick the one
     that maximises F1 on a held-out validation split, then bake that
     threshold into the served model so /predict uses it instead of 0.5.

SMOTE / oversampling is discussed in README.md as an alternative — swap the
`class_weight="balanced"` approach for an imblearn Pipeline with SMOTE if you
want to compare the two on your own dataset.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = [f"feature_{i}" for i in range(10)]


def make_imbalanced_dataset():
    """
    Stand-in for a real dataset. Replace this function with your own data
    loading (e.g. pd.read_csv(...)) — everything downstream only assumes X
    is a 2D array with FEATURE_NAMES columns and y is a binary 0/1 target.
    """
    X, y = make_classification(
        n_samples=5000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.92, 0.08],  # ~8% positive class -> realistic imbalance
        flip_y=0.01,
        random_state=RANDOM_STATE,
    )
    return X, y


def find_best_threshold(y_true, y_proba):
    """Sweep thresholds 0.01-0.99 and return the one maximising F1."""
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y_true, y_proba >= t) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def main():
    X, y = make_imbalanced_dataset()
    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features, "
          f"positive rate = {y.mean():.3%}")

    # 60% train / 20% validation (threshold tuning) / 20% test (final report)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",   # <-- imbalance handling strategy #1
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)

    val_proba = clf.predict_proba(X_val_s)[:, 1]
    best_threshold, best_f1 = find_best_threshold(y_val, val_proba)
    print(f"Chosen threshold: {best_threshold:.2f} (val F1 = {best_f1:.3f}, "
          f"default-0.5 F1 = {f1_score(y_val, val_proba >= 0.5):.3f})")

    # Final evaluation on the untouched test split, using the tuned threshold
    test_proba = clf.predict_proba(X_test_s)[:, 1]
    test_pred = test_proba >= best_threshold
    report = classification_report(y_test, test_pred, output_dict=True)
    metrics = {
        "roc_auc": roc_auc_score(y_test, test_proba),
        "pr_auc": average_precision_score(y_test, test_proba),
        "f1_at_threshold": f1_score(y_test, test_pred),
        "threshold": best_threshold,
        "test_positive_rate": float(y_test.mean()),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, test_pred))

    joblib.dump(clf, MODEL_DIR / "model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(
            {
                "feature_names": FEATURE_NAMES,
                "threshold": best_threshold,
                "metrics": metrics,
                "model_type": "RandomForestClassifier",
                "imbalance_strategy": "class_weight=balanced + tuned decision threshold",
            },
            f,
            indent=2,
        )

    print(f"\nSaved model, scaler, and metadata to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
