"""
W3D5: Hyperparameter Tuning -- GridSearch & RandomSearch
------------------------------------------------------------
Covers: exhaustive GridSearchCV vs randomized RandomizedSearchCV --
how many fits each does, how long each takes, and how close
RandomizedSearchCV's result gets to GridSearchCV's for a fraction of
the compute.

Run: python w3d5_hyperparameter_tuning.py
Outputs: prints results to console, saves PNG plots to this folder.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42


def load_data():
   
    X, y = make_classification(
        n_samples=800, n_features=20, n_informative=8, n_redundant=6,
        n_classes=2, flip_y=0.05, class_sep=0.8, random_state=RANDOM_STATE
    )
    return X, y


def run_grid_search(X_train, y_train):
   
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 4],
    }
    n_combinations = 1
    for v in param_grid.values():
        n_combinations *= len(v)

    print(f"=== GridSearchCV ===")
    print(f"Total combinations: {n_combinations} (x 5 folds = {n_combinations * 5} fits)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid, cv=cv, scoring="accuracy", n_jobs=-1
    )

    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    print(f"Time taken: {elapsed:.2f}s")
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    return grid, elapsed, n_combinations * 5


def run_random_search(X_train, y_train, n_iter=20):
   
    param_distributions = {
        "n_estimators": randint(30, 200),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 6),
    }

    print(f"\n=== RandomizedSearchCV ===")
    print(f"n_iter={n_iter} (x 5 folds = {n_iter * 5} fits)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions, n_iter=n_iter, cv=cv, scoring="accuracy",
        random_state=RANDOM_STATE, n_jobs=-1
    )

    t0 = time.perf_counter()
    random_search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    print(f"Time taken: {elapsed:.2f}s")
    print(f"Best params: {random_search.best_params_}")
    print(f"Best CV accuracy: {random_search.best_score_:.4f}")
    return random_search, elapsed, n_iter * 5


def compare_efficiency(grid_result, random_result):
    grid, grid_time, grid_fits = grid_result
    random_search, random_time, random_fits = random_result

    comparison = pd.DataFrame([
        {
            "Method": "GridSearchCV",
            "Total Fits": grid_fits,
            "Time (s)": round(grid_time, 2),
            "Best CV Accuracy": round(grid.best_score_, 4),
        },
        {
            "Method": "RandomizedSearchCV",
            "Total Fits": random_fits,
            "Time (s)": round(random_time, 2),
            "Best CV Accuracy": round(random_search.best_score_, 4),
        },
    ])
    print("\n=== Efficiency Comparison ===")
    print(comparison.to_string(index=False))
    print(f"\nRandomizedSearchCV used {random_fits / grid_fits:.1%} of the fits "
          f"GridSearchCV used, and reached "
          f"{random_search.best_score_ / grid.best_score_:.1%} of GridSearchCV's "
          f"best CV accuracy.")
    return comparison


def plot_search_progress(grid, random_search, filename):
   
    grid_scores = grid.cv_results_["mean_test_score"]
    random_scores = random_search.cv_results_["mean_test_score"]

    grid_running_best = np.maximum.accumulate(grid_scores)
    random_running_best = np.maximum.accumulate(random_scores)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(grid_running_best) + 1), grid_running_best,
              label=f"GridSearchCV ({len(grid_running_best)} combos)", color="steelblue")
    plt.plot(range(1, len(random_running_best) + 1), random_running_best,
              label=f"RandomizedSearchCV ({len(random_running_best)} combos)", color="darkorange")
    plt.xlabel("Combinations Tried")
    plt.ylabel("Best CV Accuracy So Far")
    plt.title("Search Progress: GridSearchCV vs RandomizedSearchCV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved {filename}")


def final_test_evaluation(grid, random_search, X_test, y_test):
    print("\n=== Final Held-Out Test Set Evaluation ===")
    for name, search in [("GridSearchCV best model", grid), ("RandomizedSearchCV best model", random_search)]:
        y_pred = search.best_estimator_.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name}: test accuracy = {acc:.4f}")


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    grid, grid_time, grid_fits = run_grid_search(X_train, y_train)
    random_search, random_time, random_fits = run_random_search(X_train, y_train, n_iter=20)

    compare_efficiency((grid, grid_time, grid_fits), (random_search, random_time, random_fits))
    plot_search_progress(grid, random_search, "search_progress_comparison.png")
    final_test_evaluation(grid, random_search, X_test, y_test)

    print("\n=== When to Use Which ===")
    print("GridSearchCV:")
    print("  + Guaranteed to find the best combination WITHIN the grid you define")
    print("  - Cost grows multiplicatively with each added parameter/value (combinatorial explosion)")
    print("  - Wastes time exploring unpromising regions exhaustively")
    print()
    print("RandomizedSearchCV:")
    print("  + Cost is fixed (n_iter), independent of how many parameters you search")
    print("  + Can sample from continuous distributions (e.g. loguniform), not just fixed lists")
    print("  + Often finds a near-optimal result far faster than exhaustive search")
    print("  - No guarantee of finding the true best combination")
    print()
    print("Rule of thumb: use RandomizedSearchCV for wide/exploratory searches with many")
    print("parameters, then optionally follow up with a narrow GridSearchCV around the")
    print("best region it finds.")


if __name__ == "__main__":
    main()
