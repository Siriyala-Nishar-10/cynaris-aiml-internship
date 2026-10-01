"""
W2D4: Train/Test Split & Cross-Validation + Feature Scaling
-------------------------------------------------------------
Goal: Show why distance-based (KNN) and gradient-based (SGDClassifier)
models break/underperform without scaling, and compare StandardScaler,
MinMaxScaler, and RobustScaler.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def load_data():
    """Wine dataset: 13 numeric features on very different scales
    (e.g. 'proline' ranges in the hundreds, 'hue' is < 2) -- a good
    stress test for scaling.
    """
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y


def train_test_split_demo(X, y):
    """Stratified split so class proportions are preserved in both sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate_with_cv(model, scaler, X_train, y_train, cv_folds=5):
    """Build a Pipeline(scaler -> model) so scaling is refit on each CV
    fold's training data only -- avoids data leakage from the fold's
    validation split into the scaler's fit.
    """
    steps = [("model", model)] if scaler is None else [("scaler", scaler), ("model", model)]
    pipe = Pipeline(steps)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy")
    return scores.mean(), scores.std()


def run_comparison(X_train, y_train):
    scalers = {
        "No Scaling": None,
        "StandardScaler": StandardScaler(),   # mean=0, std=1 -- best default, assumes roughly Gaussian-ish features
        "MinMaxScaler": MinMaxScaler(),       # squashes to [0,1] -- good for bounded features, sensitive to outliers
        "RobustScaler": RobustScaler(),       # uses median/IQR -- best when outliers are present
    }

    models = {
        "KNN (distance-based)": KNeighborsClassifier(n_neighbors=5),
        "SGDClassifier (gradient-based)": SGDClassifier(random_state=RANDOM_STATE, max_iter=1000),
    }

    results = []
    for model_name, model in models.items():
        for scaler_name, scaler in scalers.items():
            mean_acc, std_acc = evaluate_with_cv(model, scaler, X_train, y_train)
            results.append({
                "Model": model_name,
                "Scaler": scaler_name,
                "CV Accuracy (mean)": round(mean_acc, 4),
                "CV Accuracy (std)": round(std_acc, 4),
            })

    return pd.DataFrame(results)


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split_demo(X, y)

    results_df = run_comparison(X_train, y_train)
    print("\n=== Cross-Validation Results (5-fold, on training set) ===")
    print(results_df.to_string(index=False))

    print("\n=== Takeaway ===")
    print("KNN accuracy should jump sharply once scaling is applied, since it")
    print("relies on Euclidean distance and large-magnitude features (like")
    print("'proline') dominate the distance calculation otherwise.")
    print("SGDClassifier should also improve/stabilize with scaling, since")
    print("gradient descent converges much better when features share a scale.")


if __name__ == "__main__":
    main()
