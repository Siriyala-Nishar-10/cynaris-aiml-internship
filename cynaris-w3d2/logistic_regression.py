"""
W3D2: Logistic Regression & Classification
--------------------------------------------
Covers: sigmoid function, binary classification with LogisticRegression,
classification metrics (accuracy/precision/recall/F1/confusion matrix/ROC-AUC),
decision boundary visualization, and multi-class classification.

Run: python w3d2_logistic_regression.py
Outputs: prints results to console, saves 2 PNG plots to this folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

RANDOM_STATE = 42


def sigmoid_demo():
    """Sigmoid squashes any real number into (0, 1) -- this is what
    turns a linear combination of features into a probability.
    """
    z = np.linspace(-10, 10, 200)
    sigma = 1 / (1 + np.exp(-z))

    plt.figure(figsize=(6, 4))
    plt.plot(z, sigma, color="steelblue")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("Sigmoid Function: sigma(z) = 1 / (1 + e^-z)")
    plt.xlabel("z (linear combination of features)")
    plt.ylabel("sigma(z) = P(class = 1)")
    plt.tight_layout()
    plt.savefig("sigmoid_function.png", dpi=120)
    plt.close()
    print("Saved sigmoid_function.png")


def binary_classification_demo():
    """Binary classification: benign vs malignant tumor, using the
    breast cancer dataset. Scaling matters here too, since LogisticRegression
    is gradient-based (see W2D4).
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target  # 0 = malignant, 1 = benign

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]  # P(class = 1)

    print("\n=== Binary Classification: Breast Cancer Dataset ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # Top 5 coefficients by absolute value -- shows which features push the
    # decision most strongly toward "benign" (positive) or "malignant" (negative)
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False)
    print("\nTop 5 features by |coefficient|:")
    print(coef_df.head(5).to_string(index=False))

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", label=f"ROC curve (AUC = {roc_auc_score(y_test, y_proba):.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Breast Cancer Classification")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=120)
    plt.close()
    print("Saved roc_curve.png")


def decision_boundary_and_multiclass_demo():
    """Multi-class classification (3 iris species) using only 2 features
    so the decision boundary can be visualized directly. LogisticRegression
    handles multi-class natively via multinomial (softmax) generalization.
    """
    data = load_iris()
    X = data.data[:, :2]  # sepal length, sepal width -- for 2D plotting
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("\n=== Multi-class Classification: Iris (3 species, 2 features) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Number of decision boundaries (one-vs-rest): {model.coef_.shape[0]}")

    # Plot decision boundaries over a mesh grid
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    scatter = plt.scatter(
        X_train_scaled[:, 0], X_train_scaled[:, 1],
        c=y_train, cmap="viridis", edgecolor="k", s=40
    )
    plt.xlabel("Sepal Length (scaled)")
    plt.ylabel("Sepal Width (scaled)")
    plt.title("Logistic Regression Decision Boundaries - Iris (3 classes)")
    plt.legend(handles=scatter.legend_elements()[0], labels=list(data.target_names))
    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=120)
    plt.close()
    print("Saved decision_boundary.png")


def main():
    sigmoid_demo()
    binary_classification_demo()
    decision_boundary_and_multiclass_demo()

    print("\n=== Takeaway ===")
    print("Logistic Regression predicts P(class) via sigmoid(linear combination),")
    print("not a raw numeric value like Linear Regression. The 0.5 threshold on")
    print("that probability creates a linear decision boundary. For 3+ classes,")
    print("scikit-learn generalizes this via multinomial (softmax) logistic")
    print("regression, giving one boundary per class pair as seen in the plot.")


if __name__ == "__main__":
    main()
