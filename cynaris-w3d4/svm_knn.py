"""
W3D4: SVM & KNN -- When to Use What
--------------------------------------
Covers: SVM (linear + RBF kernel) vs KNN classification, decision boundary
visualization, hyperparameter tuning (C/gamma for SVM, n_neighbors for
KNN), and a practical comparison of when to reach for each.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42


def load_data():
   
    X, y = make_moons(n_samples=400, noise=0.25, random_state=RANDOM_STATE)
    return X, y


def scale_data(X_train, X_test):
    
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


def compare_kernels(X_train, X_test, y_train, y_test):
   
    print("=== SVM: Linear vs RBF Kernel ===")
    results = []
    for kernel in ["linear", "rbf"]:
        svm = SVC(kernel=kernel, random_state=RANDOM_STATE)
        svm.fit(X_train, y_train)
        acc = accuracy_score(y_test, svm.predict(X_test))
        results.append({"Kernel": kernel, "Test Accuracy": acc})
        print(f"{kernel:8s} kernel -> test accuracy: {acc:.4f}")
    return pd.DataFrame(results)


def tune_svm(X_train, y_train):
  
    print("\n=== Tuning SVM (RBF kernel) via GridSearchCV ===")
    param_grid = {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, "scale"]}
    grid = GridSearchCV(SVC(kernel="rbf", random_state=RANDOM_STATE), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_


def tune_knn(X_train, y_train):
    
    print("\n=== Tuning KNN via GridSearchCV ===")
    param_grid = {"n_neighbors": [3, 5, 7, 9, 11, 15], "weights": ["uniform", "distance"]}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_


def plot_decision_boundaries(models_dict, X_train, y_train, filename):
    
    fig, axes = plt.subplots(1, len(models_dict), figsize=(6 * len(models_dict), 5))
    if len(models_dict) == 1:
        axes = [axes]

    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    for ax, (name, model) in zip(axes, models_dict.items()):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolor="k", s=25)
        ax.set_title(name)
        ax.set_xlabel("Feature 1 (scaled)")
        ax.set_ylabel("Feature 2 (scaled)")

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved {filename}")


def timing_comparison(X_train, X_test, y_train, best_svm, best_knn):
   
    print("\n=== Training/Prediction Time Comparison ===")

    t0 = time.perf_counter()
    best_svm.fit(X_train, y_train)
    svm_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    best_svm.predict(X_test)
    svm_predict_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    best_knn.fit(X_train, y_train)
    knn_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    best_knn.predict(X_test)
    knn_predict_time = time.perf_counter() - t0

    timing_df = pd.DataFrame([
        {"Model": "SVM (RBF, tuned)", "Train Time (s)": svm_train_time, "Predict Time (s)": svm_predict_time},
        {"Model": "KNN (tuned)", "Train Time (s)": knn_train_time, "Predict Time (s)": knn_predict_time},
    ])
    print(timing_df.to_string(index=False))
    return timing_df


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    kernel_results = compare_kernels(X_train_scaled, X_test_scaled, y_train, y_test)

    best_svm = tune_svm(X_train_scaled, y_train)
    best_knn = tune_knn(X_train_scaled, y_train)

    svm_acc = accuracy_score(y_test, best_svm.predict(X_test_scaled))
    knn_acc = accuracy_score(y_test, best_knn.predict(X_test_scaled))

    print("\n=== Final Comparison: Tuned SVM vs Tuned KNN ===")
    final_results = pd.DataFrame([
        {"Model": "SVM (RBF, tuned)", "Test Accuracy": svm_acc},
        {"Model": "KNN (tuned)", "Test Accuracy": knn_acc},
    ])
    print(final_results.to_string(index=False))

    # Linear vs RBF SVM boundary comparison
    linear_svm = SVC(kernel="linear", random_state=RANDOM_STATE).fit(X_train_scaled, y_train)
    plot_decision_boundaries(
        {"Linear Kernel SVM": linear_svm, "RBF Kernel SVM": best_svm},
        X_train_scaled, y_train, "svm_kernel_comparison.png"
    )

    # Final SVM vs KNN boundary comparison
    plot_decision_boundaries(
        {"SVM (RBF, tuned)": best_svm, "KNN (tuned)": best_knn},
        X_train_scaled, y_train, "svm_vs_knn_boundaries.png"
    )

    timing_comparison(X_train_scaled, X_test_scaled, y_train, best_svm, best_knn)

    print("\n=== When to Use What ===")
    print("SVM (RBF kernel):")
    print("  + Works well in high-dimensional spaces, robust with clear margins")
    print("  + Fast predictions (decision depends only on support vectors)")
    print("  - Slower to train on large datasets, less interpretable")
    print("  - Requires tuning C and gamma carefully")
    print()
    print("KNN:")
    print("  + Simple, intuitive, no training phase (\"lazy learner\")")
    print("  + Naturally handles multi-class and non-linear boundaries")
    print("  - Prediction is slow on large datasets (scans all neighbors)")
    print("  - Suffers in high dimensions (curse of dimensionality)")
    print("  - Sensitive to irrelevant/unscaled features and choice of k")


if __name__ == "__main__":
    main()
