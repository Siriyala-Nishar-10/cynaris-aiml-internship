"""
sentiment_classifier.py

W4D5 Capstone: Sentiment Classifier — binary (positive/negative) text
classification comparing Logistic Regression vs Random Forest.

Practical tasks covered:
  1. Train LogisticRegression on a binary classification dataset,
     print classification_report.
  2. Plot confusion matrix and ROC-AUC curve.
  3. Try RandomForestClassifier. Compare accuracy/precision/recall vs
     Logistic Regression.

Dataset: a synthetically generated but realistic-looking set of short
product/review sentences built from positive and negative phrase templates
with randomised subjects and noise words. This keeps the capstone fully
offline and reproducible while still exercising real NLP preprocessing
(TF-IDF) rather than hand-crafted numeric features.
Swap `build_dataset()` for `pd.read_csv("your_reviews.csv")` to run this
against a real dataset — everything downstream is unchanged as long as you
end up with a DataFrame with "text" and "label" (0/1) columns.
"""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

OUT_DIR = Path(__file__).parent
N_SAMPLES = 1200

POSITIVE_TEMPLATES = [
    "I absolutely love this {subject}, it exceeded my expectations",
    "This {subject} is fantastic and works perfectly every time",
    "Great {subject}, would definitely buy again",
    "The {subject} is amazing, best purchase I've made this year",
    "Really happy with this {subject}, excellent quality and value",
    "This {subject} performs wonderfully and looks great too",
    "Superb {subject}, exactly what I was looking for",
    "I'm impressed by how reliable this {subject} is",
    "Highly recommend this {subject} to anyone",
    "The {subject} arrived quickly and works beautifully",
]

NEGATIVE_TEMPLATES = [
    "I hate this {subject}, it broke after one day",
    "This {subject} is terrible and a complete waste of money",
    "Very disappointed with this {subject}, would not recommend",
    "The {subject} stopped working within a week",
    "Poor quality {subject}, does not match the description",
    "This {subject} is awful and feels cheaply made",
    "Worst {subject} I have ever purchased",
    "The {subject} arrived damaged and customer service was unhelpful",
    "Not worth the price, this {subject} is a disappointment",
    "I regret buying this {subject}, it barely functions",
]

SUBJECTS = [
    "phone", "laptop", "blender", "headphones", "backpack", "watch",
    "camera", "keyboard", "chair", "speaker", "tablet", "charger",
    "monitor", "mouse", "vacuum", "toaster", "router", "printer",
]

NOISE_SUFFIXES = [
    "", " Shipping was fast.", " Packaging was fine.",
    " I bought it last month.", " My friend recommended it.",
    " It was on sale.", " I use it every day.",
]

# Negation-based examples deliberately included to stress-test bag-of-words
# style models: the sentiment word ("good"/"bad") is the opposite of the
# true label once "not" is applied. TF-IDF has no notion of word order, so
# these are genuinely hard for both models and keep accuracy realistic
# instead of a trivial 100%.
NEGATION_TEMPLATES = [
    ("This {subject} is not bad at all, quite decent actually", 1),
    ("Honestly this {subject} is not good, don't buy it", 0),
    ("The {subject} isn't great but it's not terrible either", 1),
    ("I wouldn't say this {subject} is amazing, it's underwhelming", 0),
    ("Not the worst {subject} I've owned, works okay", 1),
    ("This {subject} is not what I expected, quite disappointing", 0),
]

LABEL_NOISE_RATE = 0.05  # simulates imperfect real-world annotation


def build_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    rows = []
    n_negation = int(n_samples * 0.12)
    n_regular = n_samples - n_negation

    for _ in range(n_regular):
        label = random.randint(0, 1)
        templates = POSITIVE_TEMPLATES if label == 1 else NEGATIVE_TEMPLATES
        text = random.choice(templates).format(subject=random.choice(SUBJECTS))
        text += random.choice(NOISE_SUFFIXES)
        rows.append({"text": text, "label": label})

    for _ in range(n_negation):
        template, label = random.choice(NEGATION_TEMPLATES)
        text = template.format(subject=random.choice(SUBJECTS))
        text += random.choice(NOISE_SUFFIXES)
        rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Inject label noise to mimic real annotation error
    flip_mask = np.random.rand(len(df)) < LABEL_NOISE_RATE
    df.loc[flip_mask, "label"] = 1 - df.loc[flip_mask, "label"]

    return df


def evaluate_model(name, model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    print(f"\n{'=' * 60}\n{name} — classification_report\n{'=' * 60}")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return y_pred, y_proba, metrics


def main():
    # 1. Build / load dataset
    df = build_dataset()
    print(f"Dataset: {len(df)} samples, "
          f"positive rate = {df['label'].mean():.1%}")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # 2. Logistic Regression
    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg.fit(X_train_vec, y_train)
    lr_pred, lr_proba, lr_metrics = evaluate_model(
        "Logistic Regression", logreg, X_test_vec, y_test
    )

    # 3. Random Forest, for comparison
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train_vec, y_train)
    rf_pred, rf_proba, rf_metrics = evaluate_model(
        "Random Forest", rf, X_test_vec, y_test
    )

    # --- Confusion matrices (side by side) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, lr_pred), display_labels=["negative", "positive"]
    ).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Logistic Regression")
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, rf_pred), display_labels=["negative", "positive"]
    ).plot(ax=axes[1], colorbar=False, cmap="Greens")
    axes[1].set_title("Random Forest")
    fig.suptitle("Confusion Matrices — Sentiment Classifier")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # --- ROC-AUC curves (overlaid) ---
    fig, ax = plt.subplots(figsize=(6, 5.5))
    RocCurveDisplay.from_predictions(y_test, lr_proba, name="Logistic Regression", ax=ax)
    RocCurveDisplay.from_predictions(y_test, rf_proba, name="Random Forest", ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curve — Logistic Regression vs Random Forest")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curve_comparison.png", dpi=150)
    plt.close(fig)

    # --- Comparison table ---
    comparison = pd.DataFrame(
        [
            {"model": "Logistic Regression", **lr_metrics},
            {"model": "Random Forest", **rf_metrics},
        ]
    ).round(4)
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    print(f"\n{'=' * 60}\nModel comparison\n{'=' * 60}")
    print(comparison.to_string(index=False))

    with open(OUT_DIR / "results_summary.json", "w") as f:
        json.dump(
            {"logistic_regression": lr_metrics, "random_forest": rf_metrics},
            f,
            indent=2,
        )

    print(f"\nSaved confusion_matrices.png, roc_curve_comparison.png, "
          f"model_comparison.csv, results_summary.json")


if __name__ == "__main__":
    main()
