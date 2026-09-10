"""
W2D3: Handling Imbalanced Data — SMOTE
Task: demonstrate why class imbalance is a problem, apply SMOTE to
rebalance, and show why accuracy alone is a misleading metric.

Author: Siriyala Nishar
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from imblearn.over_sampling import SMOTE


def inspect_class_balance(df: pd.DataFrame) -> None:
    """Report the class distribution to make the imbalance explicit."""
    counts = df["is_fraud"].value_counts()
    pct = df["is_fraud"].value_counts(normalize=True) * 100
    print("=== Class Balance ===")
    print(f"Legitimate (0): {counts[0]} ({pct[0]:.1f}%)")
    print(f"Fraud (1):      {counts[1]} ({pct[1]:.1f}%)")


def train_baseline_no_smote(X_train, X_test, y_train, y_test) -> dict:
    """Train a baseline model WITHOUT addressing imbalance, to show
    why accuracy alone is misleading on imbalanced data.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n=== Baseline (NO SMOTE) ===")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.3f}")
    print(f"Precision: {precision_score(y_test, preds, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.3f}")
    print(f"F1:        {f1_score(y_test, preds, zero_division=0):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print(
        "\nNote: high accuracy here can be misleading — a model that "
        "predicts 'not fraud' for almost everything still scores well "
        "on accuracy, because 95% of the data genuinely isn't fraud."
    )
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }


def apply_smote_and_retrain(X_train, X_test, y_train, y_test) -> dict:
    """Apply SMOTE to the TRAINING set only (never the test set — that
    would leak synthetic patterns into evaluation), then retrain and
    compare metrics.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"\n=== After SMOTE ===")
    print(f"Training set before: {y_train.value_counts().to_dict()}")
    print(f"Training set after:  {y_resampled.value_counts().to_dict()}")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_resampled, y_resampled)
    preds = model.predict(X_test)

    print("\n=== With SMOTE ===")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.3f}")
    print(f"Precision: {precision_score(y_test, preds, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.3f}")
    print(f"F1:        {f1_score(y_test, preds, zero_division=0):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }


def plot_comparison(before: dict, after: dict, out_dir: Path) -> None:
    """Plot before/after metrics side by side."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width/2 for i in x], [before[m] for m in metrics], width, label="No SMOTE")
    plt.bar([i + width/2 for i in x], [after[m] for m in metrics], width, label="With SMOTE")
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("Model Performance: Before vs After SMOTE")
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "smote_comparison.png", dpi=120)
    plt.close()
    print("\nSaved smote_comparison.png")


def main() -> None:
    csv_path = Path("transaction_fraud.csv")
    out_dir = Path(".")

    df = pd.read_csv(csv_path)
    inspect_class_balance(df)

    feature_cols = [
        "transaction_amount", "transaction_hour",
        "account_age_days", "num_prior_transactions",
    ]
    X = df[feature_cols]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    before = train_baseline_no_smote(X_train, X_test, y_train, y_test)
    after = apply_smote_and_retrain(X_train, X_test, y_train, y_test)
    plot_comparison(before, after, out_dir)

    print("\n=== Summary ===")
    print(f"Recall improved from {before['recall']:.3f} to {after['recall']:.3f}")
    print("(Recall matters most here: missing actual fraud is costlier")
    print("than a false alarm, so improving recall — even at some")
    print("precision cost — is usually the right trade-off for fraud detection.)")


if __name__ == "__main__":
    main()
