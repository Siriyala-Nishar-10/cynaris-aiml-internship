"""
W4D2: Bias-Variance Tradeoff & Regularisation (L1 vs L2)
--------------------------------------------------------------
Covers: the classic bias-variance tradeoff via polynomial regression
complexity, and how L1 (Lasso) vs L2 (Ridge) regularization control
that tradeoff differently -- Lasso can zero out coefficients entirely
(feature selection), Ridge only shrinks them toward zero.

Run: python w4d2_bias_variance_regularization.py
Outputs: prints results to console, saves PNG plots to this folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

RANDOM_STATE = 42
rng = np.random.RandomState(RANDOM_STATE)


def generate_data(n_samples=80, noise=3.0):
    """True relationship is a simple cubic curve. We'll fit polynomials
    of increasing degree to it -- low degree underfits (high bias),
    very high degree overfits to the noise (high variance).
    """
    X = np.sort(rng.uniform(-3, 3, n_samples))
    y_true = 0.5 * X**3 - 2 * X**2 + X + 5
    y = y_true + rng.normal(0, noise, n_samples)
    return X.reshape(-1, 1), y, y_true


def bias_variance_via_polynomial_degree(X, y, X_test, y_test):
    """Sweep polynomial degree from underfitting through overfitting,
    tracking train vs test MSE at each degree -- the classic
    bias-variance U-curve.
    """
    print("=== Bias-Variance Tradeoff: Polynomial Degree Sweep ===")
    degrees = [1, 2, 3, 4, 6, 9, 12, 15]
    train_errors, test_errors = [], []

    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        train_mse = mean_squared_error(y, model.predict(X))
        test_mse = mean_squared_error(y_test, model.predict(X_test))
        train_errors.append(train_mse)
        test_errors.append(test_mse)
        regime = "underfitting" if degree <= 2 else ("overfitting" if degree >= 9 else "near-optimal")
        print(f"Degree {degree:2d}: train MSE={train_mse:7.2f}, test MSE={test_mse:7.2f}  ({regime})")

    return degrees, train_errors, test_errors


def plot_bias_variance_curve(degrees, train_errors, test_errors, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, train_errors, "o-", color="steelblue", label="Train MSE")
    plt.plot(degrees, test_errors, "o-", color="darkorange", label="Test MSE")
    plt.axvspan(0.5, 2.5, alpha=0.1, color="blue", label="Underfitting (high bias)")
    plt.axvspan(8.5, 15.5, alpha=0.1, color="red", label="Overfitting (high variance)")
    plt.xlabel("Polynomial Degree (model complexity)")
    plt.ylabel("Mean Squared Error")
    plt.title("Bias-Variance Tradeoff: Train vs Test Error by Model Complexity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved {filename}")


def plot_fit_examples(X, y, X_test, y_test, filename):
    """Visually show underfit, good fit, and overfit curves against
    the actual data points.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    configs = [(1, "Underfitting (degree=1)"), (3, "Good Fit (degree=3)"), (15, "Overfitting (degree=15)")]

    X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    for ax, (degree, title) in zip(axes, configs):
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        y_plot = model.predict(X_plot)

        ax.scatter(X, y, color="gray", s=20, alpha=0.6, label="Train data")
        ax.plot(X_plot, y_plot, color="crimson", linewidth=2, label="Model fit")
        ax.set_title(title)
        ax.set_ylim(y.min() - 10, y.max() + 10)
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved {filename}")


def l1_vs_l2_demo():
    """Build a dataset where only a few features actually matter, then
    compare plain LinearRegression, Ridge (L2), and Lasso (L1) on how
    they handle the irrelevant features' coefficients.
    """
    print("\n=== L1 (Lasso) vs L2 (Ridge) Regularization ===")
    n_samples, n_features, n_informative = 100, 20, 5

    X = rng.normal(0, 1, (n_samples, n_features))
    true_coef = np.zeros(n_features)
    true_coef[:n_informative] = rng.uniform(2, 5, n_informative)  # only first 5 features matter
    y = X @ true_coef + rng.normal(0, 2, n_samples)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear Regression (no penalty)": LinearRegression(),
        "Ridge (L2, alpha=1.0)": Ridge(alpha=1.0),
        "Lasso (L1, alpha=0.1)": Lasso(alpha=0.1),
    }

    coef_data = {"feature": feature_names, "true_coef": true_coef}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        test_mse = mean_squared_error(y_test, model.predict(X_test_s))
        n_zero = np.sum(np.abs(model.coef_) < 1e-3)
        print(f"{name}: test MSE={test_mse:.3f}, coefficients set to ~0: {n_zero}/{n_features}")
        coef_data[name] = model.coef_

    coef_df = pd.DataFrame(coef_data)
    print("\nCoefficient comparison (first 8 features shown):")
    print(coef_df.head(8).round(3).to_string(index=False))

    return coef_df


def plot_coefficient_comparison(coef_df, filename):
    """Bar chart comparing coefficient magnitudes across models --
    shows Lasso zeroing out irrelevant (feature_5 onward) features
    while Ridge just shrinks everything toward zero without hitting it.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(coef_df))
    width = 0.2

    ax.bar(x - 1.5 * width, coef_df["true_coef"], width, label="True coefficient", color="black", alpha=0.6)
    ax.bar(x - 0.5 * width, coef_df["Linear Regression (no penalty)"], width, label="Linear Regression")
    ax.bar(x + 0.5 * width, coef_df["Ridge (L2, alpha=1.0)"], width, label="Ridge (L2)")
    ax.bar(x + 1.5 * width, coef_df["Lasso (L1, alpha=0.1)"], width, label="Lasso (L1)")

    ax.set_xticks(x)
    ax.set_xticklabels(coef_df["feature"], rotation=45, ha="right")
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Coefficient Comparison: True vs Linear vs Ridge vs Lasso\n(features 5-19 are noise -- true coefficient is 0)")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved {filename}")


def regularization_strength_sweep(filename):
    """Sweep Ridge's alpha to show the bias-variance tradeoff directly
    through a regularization hyperparameter: alpha=0 behaves like plain
    linear regression (can overfit / high variance); very high alpha
    shrinks everything toward 0 (underfits / high bias).
    """
    print("\n=== Regularization Strength (alpha) vs Bias-Variance ===")
    n_samples, n_features, n_informative = 40, 30, 4
    X = rng.normal(0, 1, (n_samples, n_features))
    true_coef = np.zeros(n_features)
    true_coef[:n_informative] = rng.uniform(2, 4, n_informative)
    y = X @ true_coef + rng.normal(0, 4, n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    alphas = np.logspace(-2, 3, 20)
    train_mses, test_mses = [], []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train_s, y_train)
        train_mses.append(mean_squared_error(y_train, model.predict(X_train_s)))
        test_mses.append(mean_squared_error(y_test, model.predict(X_test_s)))

    best_idx = np.argmin(test_mses)
    print(f"Best alpha: {alphas[best_idx]:.4f} (lowest test MSE = {test_mses[best_idx]:.3f})")
    print(f"alpha=0.01 (near-unregularized): test MSE={test_mses[0]:.3f}")
    print(f"alpha=1000 (heavily regularized): test MSE={test_mses[-1]:.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, train_mses, "o-", color="steelblue", label="Train MSE")
    plt.plot(alphas, test_mses, "o-", color="darkorange", label="Test MSE")
    plt.axvline(alphas[best_idx], color="green", linestyle="--", label=f"Best alpha={alphas[best_idx]:.3f}")
    plt.xscale("log")
    plt.xlabel("Ridge alpha (regularization strength, log scale)")
    plt.ylabel("Mean Squared Error")
    plt.title("Regularization Strength vs Bias-Variance (Ridge)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved {filename}")


def main():
    X, y, y_true = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

    degrees, train_errors, test_errors = bias_variance_via_polynomial_degree(X_train, y_train, X_test, y_test)
    plot_bias_variance_curve(degrees, train_errors, test_errors, "bias_variance_curve.png")
    plot_fit_examples(X_train, y_train, X_test, y_test, "underfit_good_overfit_examples.png")

    coef_df = l1_vs_l2_demo()
    plot_coefficient_comparison(coef_df, "l1_vs_l2_coefficients.png")

    regularization_strength_sweep("ridge_alpha_sweep.png")

    print("\n=== Takeaway ===")
    print("Bias-variance tradeoff: too-simple models (low polynomial degree,")
    print("or heavy regularization) underfit -- high train AND test error")
    print("(high bias). Too-complex models (high degree, no regularization)")
    print("overfit -- near-zero train error but high test error (high variance).")
    print()
    print("L1 (Lasso) adds |coefficient| penalty -> can shrink coefficients")
    print("exactly to zero, performing automatic feature selection.")
    print("L2 (Ridge) adds coefficient^2 penalty -> shrinks coefficients toward")
    print("zero but rarely exactly to zero, keeping all features but with")
    print("reduced influence. Both trade some bias for reduced variance.")


if __name__ == "__main__":
    main()
