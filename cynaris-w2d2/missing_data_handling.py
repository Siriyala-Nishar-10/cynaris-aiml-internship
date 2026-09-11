"""
W2D2: Feature Scaling & Selection — Missing Data Handling
Task: apply and compare imputation strategies (simple, KNN, iterative)
and document when each is appropriate.

Author: Siriyala Nishar
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


def inspect_missingness(df: pd.DataFrame) -> None:
    """Report missing value counts and percentages per column.

    Why: the right imputation strategy depends on HOW MUCH is missing
    and WHY — a quick inspection is the first step, not a guess.
    """
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(1)
    summary = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    print("=== Missingness Summary ===")
    print(summary[summary["missing_count"] > 0])


def simple_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Apply SimpleImputer: mean for one numeric column, median for
    another, most_frequent for the categorical column.

    Why mean vs median: 'age' has no major outliers, so mean is fine.
    'annual_income' can have skew/outliers, so median is safer. For
    the categorical 'department', most_frequent (mode) is the only
    sensible simple strategy.

    Limitation: SimpleImputer treats every row independently — it
    ignores relationships between columns, so it's fast but naive.
    """
    df = df.copy()

    age_imputer = SimpleImputer(strategy="mean")
    df["age_imputed_simple"] = age_imputer.fit_transform(df[["age"]])

    income_imputer = SimpleImputer(strategy="median")
    df["income_imputed_simple"] = income_imputer.fit_transform(df[["annual_income"]])

    dept_imputer = SimpleImputer(strategy="most_frequent")
    df["department_imputed_simple"] = dept_imputer.fit_transform(
        df[["department"]]
    ).ravel()

    print("\n=== Simple Imputation applied ===")
    print(f"age: filled with mean = {age_imputer.statistics_[0]:.1f}")
    print(f"annual_income: filled with median = {income_imputer.statistics_[0]:.1f}")
    print(f"department: filled with mode = {dept_imputer.statistics_[0]}")
    return df


def knn_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Apply KNNImputer on numeric columns together.

    Why: KNN imputation fills a missing value based on the average of
    the k most SIMILAR rows (by other numeric features) — more
    informed than SimpleImputer since it uses relationships between
    columns, at the cost of being slower and needing scaled numeric
    input for 'similarity' to be meaningful.
    """
    numeric_cols = ["age", "annual_income", "years_experience", "credit_score"]
    imputer = KNNImputer(n_neighbors=5)
    imputed_values = imputer.fit_transform(df[numeric_cols])
    result = df.copy()
    for i, col in enumerate(numeric_cols):
        result[f"{col}_imputed_knn"] = imputed_values[:, i]

    print("\n=== KNN Imputation applied (k=5) ===")
    print("Filled using the 5 most similar rows per missing value.")
    return result


def iterative_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Apply IterativeImputer (MICE-style) on numeric columns together.

    Why: models each feature with missing values as a function of the
    other features, iterating until estimates stabilize — the most
    sophisticated of the three, capturing multivariate relationships
    KNN and SimpleImputer miss. Trade-off: slowest, and can be
    unstable with very small or noisy datasets.
    """
    numeric_cols = ["age", "annual_income", "years_experience", "credit_score"]
    imputer = IterativeImputer(max_iter=10, random_state=42)
    imputed_values = imputer.fit_transform(df[numeric_cols])
    result = df.copy()
    for i, col in enumerate(numeric_cols):
        result[f"{col}_imputed_iterative"] = imputed_values[:, i]

    print("\n=== Iterative Imputation applied (MICE-style) ===")
    print("Each feature modelled as a function of the others, iteratively.")
    return result


def compare_strategies(df: pd.DataFrame) -> None:
    """Compare imputed values for 'years_experience' (the block-missing
    column) across all 3 strategies, since that's the most informative
    comparison — a large missing block is where method choice matters most.
    """
    sample_idx = df[df["years_experience"].isna()].index[:5]
    comparison = pd.DataFrame({
        "original (NaN)": df.loc[sample_idx, "years_experience"],
    })
    print("\n=== Strategy Comparison on 'years_experience' (5 sample rows) ===")
    print(comparison)
    print("(Full imputed values available in the exported CSV columns)")


def main() -> None:
    csv_path = Path("employee_missing_data.csv")
    df = pd.read_csv(csv_path)

    inspect_missingness(df)
    df = simple_imputation(df)
    df = knn_imputation(df)
    df = iterative_imputation(df)
    compare_strategies(df)

    df.to_csv("employee_data_imputed.csv", index=False)
    print("\nSaved employee_data_imputed.csv with all imputation strategies.")


if __name__ == "__main__":
    main()
