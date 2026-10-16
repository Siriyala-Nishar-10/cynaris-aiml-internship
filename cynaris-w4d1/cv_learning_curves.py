"""
W4D1: Model Evaluation Metrics -- Precision, Recall, AUC
   + K-Fold CV, Stratified CV, and Learning Curves
------------------------------------------------------------
Why a single train/test split isn't enough: one split can get lucky or
unlucky, giving a misleading picture of how a model really performs.
K-fold (and stratified k-fold) CV averages over multiple splits for a
more reliable estimate. Learning curves then show WHY a model is
under/overperforming -- by tracking train vs validation score as
training set size grows.

Run: python w4d1_cv_learning_curves.py
Outputs: prints results to console, saves PNG plots to this folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, learning_curve, train_test_split
)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

RANDOM_STATE = 42


def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y


def why_single_split_is_unreliable(X, y, model):
  
    print("=== Why One Train/Test Split Isn't Enough ===")
    scores = []
    for seed in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model.fit(X_train_s, y_train)
        scores.append(model.score(X_test_s, y_test))

    scores = np.array(scores)
    print(f"10 different random splits -> accuracy range: "
          f"{scores.min():.4f} to {scores.max():.4f} "
          f"(spread of {scores.max() - scores.min():.4f})")
    print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
    print("A single split could have landed anywhere in that range --")
    print("which is exactly why we average over many splits with CV.\n")
    return scores


def compare_kfold_vs_stratified(X, y, model):
   
    print("=== KFold vs StratifiedKFold ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    kfold_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring="accuracy")

    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    skfold_scores = cross_val_score(model, X_scaled, y, cv=skfold, scoring="accuracy")

    print(f"KFold:           mean={kfold_scores.mean():.4f}, std={kfold_scores.std():.4f}, "
          f"fold scores={np.round(kfold_scores, 4)}")
    print(f"StratifiedKFold: mean={skfold_scores.mean():.4f}, std={skfold_scores.std():.4f}, "
          f"fold scores={np.round(skfold_scores, 4)}")
    lower_variance = "StratifiedKFold" if skfold_scores.std() < kfold_scores.std() else "KFold"
    print(f"On this run, {lower_variance} showed lower variance across folds.")
    print("In theory StratifiedKFold should have equal or lower variance, since it")
    print("guarantees every fold matches the overall class balance -- but with")
    print("shuffle=True on a dataset that's only mildly imbalanced (37%/63% here),")
    print("plain KFold can also land on well-balanced folds by chance. The real")
    print("benefit of stratification shows up more reliably on smaller or more")
    print("heavily imbalanced datasets, where an unlucky KFold split could leave")
    print("a fold with very few (or zero) minority-class examples.\n")
    return kfold_scores, skfold_scores


def full_metrics_with_cv(X, y, models_dict):
    
    print("=== Precision / Recall / F1 / ROC-AUC via 5-Fold Stratified CV ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for name, model in models_dict.items():
        precision = cross_val_score(model, X_scaled, y, cv=skfold, scoring="precision").mean()
        recall = cross_val_score(model, X_scaled, y, cv=skfold, scoring="recall").mean()
        f1 = cross_val_score(model, X_scaled, y, cv=skfold, scoring="f1").mean()
        auc = cross_val_score(model, X_scaled, y, cv=skfold, scoring="roc_auc").mean()
        results.append({"Model": name, "Precision": precision, "Recall": recall,
                         "F1": f1, "ROC-AUC": auc})

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return results_df


def plot_learning_curves(X, y, models_dict, filename):
   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fig, axes = plt.subplots(1, len(models_dict), figsize=(6 * len(models_dict), 5))
    if len(models_dict) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models_dict.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_scaled, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy", n_jobs=-1
        )
        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)

        ax.plot(train_sizes, train_mean, "o-", color="steelblue", label="Training score")
        ax.plot(train_sizes, val_mean, "o-", color="darkorange", label="Validation score")
        ax.set_title(name)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="lower right")
        ax.set_ylim(0.85, 1.02)

        gap = train_mean[-1] - val_mean[-1]
        diagnosis = "overfitting" if gap > 0.03 else "good fit / mild bias"
        print(f"{name}: final train={train_mean[-1]:.4f}, final val={val_mean[-1]:.4f}, "
              f"gap={gap:.4f} -> {diagnosis}")

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved {filename}")


def main():
    X, y = load_data()
    print(f"Dataset shape: {X.shape}, class balance: {np.bincount(y)}\n")

    reference_model = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    why_single_split_is_unreliable(X, y, reference_model)
    compare_kfold_vs_stratified(X, y, reference_model)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    full_metrics_with_cv(X, y, models)

    print("\n=== Learning Curves: Diagnosing Overfitting vs Underfitting ===")
    plot_learning_curves(
        X, y,
        {"Decision Tree (max_depth=5)": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
         "Decision Tree (unrestricted)": DecisionTreeClassifier(random_state=RANDOM_STATE),
         "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)},
        "learning_curves.png"
    )

    print("\n=== Takeaway ===")
    print("A single train/test split gives one noisy accuracy estimate; the")
    print("10-split experiment above shows how much that single number can")
    print("swing. K-fold CV averages over multiple splits for a stable")
    print("estimate, and StratifiedKFold keeps class balance consistent")
    print("across folds. Learning curves go a level deeper than any single")
    print("CV score: a large, persistent train-validation gap signals")
    print("overfitting (high variance), while both curves converging to a")
    print("low score signals underfitting (high bias).")


if __name__ == "__main__":
    main()
