"""
W3D1: Linear Regression — Scikit-Learn
Task: train LinearRegression, Ridge, and Lasso on house price data,
evaluate with MSE/RMSE/MAE/R², plot predicted vs actual and residuals,
and compare all 3 models.

Author: Siriyala Nishar
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_and_evaluate(model, X_train, X_test, y_train, y_test, name: str) -> dict:
    """Fit a model, predict, and compute MSE/RMSE/MAE/R²."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n=== {name} ===")
    print(f"MSE:  {mse:,.0f}")
    print(f"RMSE: {rmse:,.0f}")
    print(f"MAE:  {mae:,.0f}")
    print(f"R²:   {r2:.4f}")

    return {
        "model": name, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2,
        "predictions": preds, "fitted_model": model,
    }


def print_coefficients(model, feature_names: list[str], name: str) -> None:
    """Print model coefficients to show each feature's learned impact."""
    print(f"\n{name} coefficients:")
    for feat, coef in zip(feature_names, model.coef_):
        print(f"  {feat}: {coef:,.2f}")
    print(f"  intercept: {model.intercept_:,.2f}")


def plot_predicted_vs_actual(results: list[dict], y_test, out_dir: Path) -> None:
    """Plot predicted vs actual for all 3 models side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, res in zip(axes, results):
        ax.scatter(y_test, res["predictions"], alpha=0.5)
        lims = [y_test.min(), y_test.max()]
        ax.plot(lims, lims, "r--", label="Perfect prediction")
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title(f"{res['model']} (R²={res['r2']:.3f})")
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "predicted_vs_actual.png", dpi=120)
    plt.close()
    print("\nSaved predicted_vs_actual.png")


def plot_residuals(results: list[dict], y_test, out_dir: Path) -> None:
    """Plot residuals (actual - predicted) for all 3 models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, res in zip(axes, results):
        residuals = y_test - res["predictions"]
        ax.scatter(res["predictions"], residuals, alpha=0.5)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Predicted Price")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.set_title(f"{res['model']} Residuals")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals.png", dpi=120)
    plt.close()
    print("Saved residuals.png")


def main() -> None:
    csv_path = Path("house_prices.csv")
    out_dir = Path(".")

    df = pd.read_csv(csv_path)
    feature_cols = ["sqft", "bedrooms", "bathrooms", "age_years", "distance_to_city_km", "garage"]
    X = df[feature_cols]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features — important for Ridge/Lasso, whose penalty is
    # scale-sensitive (a feature with a larger numeric range would be
    # penalised differently than one with a smaller range, purely due
    # to units, not true importance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    lr = LinearRegression()
    results.append(train_and_evaluate(lr, X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression"))
    print_coefficients(lr, feature_cols, "Linear Regression")

    ridge = Ridge(alpha=1.0)
    results.append(train_and_evaluate(ridge, X_train_scaled, X_test_scaled, y_train, y_test, "Ridge (alpha=1.0)"))
    print_coefficients(ridge, feature_cols, "Ridge")

    lasso = Lasso(alpha=1.0)
    results.append(train_and_evaluate(lasso, X_train_scaled, X_test_scaled, y_train, y_test, "Lasso (alpha=1.0)"))
    print_coefficients(lasso, feature_cols, "Lasso")

    plot_predicted_vs_actual(results, y_test, out_dir)
    plot_residuals(results, y_test, out_dir)

    print("\n=== Comparison Table ===")
    comparison = pd.DataFrame([
        {"Model": r["model"], "MSE": r["mse"], "RMSE": r["rmse"], "MAE": r["mae"], "R2": r["r2"]}
        for r in results
    ])
    print(comparison.to_string(index=False))
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)
    print("\nSaved model_comparison.csv")


if __name__ == "__main__":
    main()
