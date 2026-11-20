
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def main():
    # Deterministic demonstration dataset
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, size=(300, 3))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] - X[:, 2] + 4.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Use SQLite instead of the deprecated filesystem backend
    tracking_db = ROOT / "mlflow.db"
    mlflow.set_tracking_uri(
        f"sqlite:///{tracking_db.as_posix()}"
    )

    mlflow.set_experiment("w12d5-production-ml-api")

    with mlflow.start_run(run_name="linear-regression-baseline"):
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("test_mse", float(mse))
        mlflow.log_metric("test_r2", float(r2))

        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

    joblib.dump(model, ARTIFACTS / "model.joblib")

    print(f"Training complete: test_mse={mse:.4f}, test_r2={r2:.4f}")
    print(f"Model saved to {ARTIFACTS / 'model.joblib'}")
    print(f"MLflow database: {tracking_db}")


if __name__ == "__main__":
    main()
