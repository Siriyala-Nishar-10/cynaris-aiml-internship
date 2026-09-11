"""
W2D5: Week 2 Project — End-to-End Preprocessing Pipeline
Applies everything from Week 2 (EDA, missing data, encoding, scaling)
on a Titanic-style dataset, using sklearn's ColumnTransformer + Pipeline
to build a single, reusable, leak-safe preprocessing pipeline.

Author: Siriyala Nishar
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def quick_eda(df: pd.DataFrame, out_dir: Path) -> None:
    """Step 1: EDA — shape, missing values, survival rate by class/sex."""
    print("=== Shape ===", df.shape)
    print("\n=== Missing values ===")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\n=== Survival rate by class ===")
    print(df.groupby("pclass")["survived"].mean().round(3))
    print("\n=== Survival rate by sex ===")
    print(df.groupby("sex")["survived"].mean().round(3))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df.groupby("pclass")["survived"].mean().plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Survival Rate by Class")
    axes[0].set_ylabel("Survival Rate")
    df.groupby("sex")["survived"].mean().plot(kind="bar", ax=axes[1], color="salmon")
    axes[1].set_title("Survival Rate by Sex")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_survival_rates.png", dpi=120)
    plt.close()
    print("\nSaved eda_survival_rates.png")


def build_pipeline() -> Pipeline:
    """Steps 2-4: build ONE pipeline combining missing-value handling,
    encoding, and scaling — via ColumnTransformer.

    Why ColumnTransformer + Pipeline (not manual step-by-step code):
    it fits all transformations on TRAINING data only, then applies
    the *same learned* transformation to test data — preventing data
    leakage (e.g. the test set's mean/median never influences imputation
    or scaling, which would happen if you fit on the whole dataset
    before splitting).
    """
    numeric_features = ["age", "fare", "sibsp", "parch"]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_features = ["sex", "embarked", "pclass"]
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return preprocessor


def main() -> None:
    csv_path = Path("titanic_like.csv")
    out_dir = Path(".")

    df = pd.read_csv(csv_path)
    quick_eda(df, out_dir)

    X = df.drop(columns=["passenger_id", "survived"])
    y = df["survived"]

    # Split BEFORE fitting any transformer — this is the leak-safe order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    preprocessor = build_pipeline()

    # Fit ONLY on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    # Transform test data using what was learned from training data only
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    print(f"\nProcessed feature count: {len(feature_names)}")
    print("Feature names:", list(feature_names))

    # Export ML-ready features
    train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    train_df["survived"] = y_train.values
    test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    test_df["survived"] = y_test.values

    train_df.to_csv(out_dir / "titanic_train_ml_ready.csv", index=False)
    test_df.to_csv(out_dir / "titanic_test_ml_ready.csv", index=False)
    print("\nSaved titanic_train_ml_ready.csv and titanic_test_ml_ready.csv")

    print("\n=== Sample of ML-ready training data ===")
    print(train_df.head())


if __name__ == "__main__":
    main()
