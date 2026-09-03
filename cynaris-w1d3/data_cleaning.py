"""
W1D3: Data Loading, Cleaning & Inspection
Task: load a raw (messy) dataset, inspect it, clean it (missing values,
duplicates, inconsistent formatting, outliers), and verify the result.

Author: Siriyala Nishar
"""

from pathlib import Path
import numpy as np
import pandas as pd


def load_and_inspect(csv_path: Path) -> pd.DataFrame:
    """Load the raw CSV and print an inspection summary.

    Why: before cleaning anything, you need to know what's actually
    wrong — missing values, duplicates, dtypes — rather than guessing.
    """
    df = pd.read_csv(csv_path)
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nHead:\n", df.head())
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully duplicate rows, keeping the first occurrence."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"\nRemoved {before - len(df)} duplicate rows")
    return df


def standardise_city_names(df: pd.DataFrame) -> pd.DataFrame:
    """Fix inconsistent casing in the city column (e.g. 'bangalore' -> 'Bangalore').

    Why: inconsistent casing makes groupby/filtering silently wrong,
    since 'bangalore' and 'Bangalore' would be treated as different groups.
    """
    df["city"] = df["city"].str.title()
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numeric values with the column median; drop rows
    with missing city (since city can't be reasonably guessed).

    Why: median is more robust to outliers than mean for imputation.
    Dropping rows with missing categorical data (vs guessing) avoids
    introducing incorrect assumptions into a field with no numeric basis.
    """
    df["age"] = df["age"].fillna(df["age"].median())
    df["monthly_income"] = df["monthly_income"].fillna(
        df["monthly_income"].median()
    )
    before = len(df)
    df = df.dropna(subset=["city"]).reset_index(drop=True)
    print(f"\nDropped {before - len(df)} rows with missing city")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with implausible age values (e.g. age > 100).

    Why: an age of 150 is a data entry error, not a valid outlier to
    keep — domain knowledge (max realistic human age) drives this rule,
    rather than a purely statistical threshold.
    """
    before = len(df)
    df = df[df["age"] <= 100].reset_index(drop=True)
    print(f"\nRemoved {before - len(df)} rows with implausible age")
    return df


def verify_clean(df: pd.DataFrame) -> None:
    """Print a final summary to confirm the dataset is clean."""
    print("\n--- Post-cleaning verification ---")
    print("Shape:", df.shape)
    print("Missing values:\n", df.isna().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Age range:", df["age"].min(), "-", df["age"].max())
    print("Unique cities:", sorted(df["city"].unique()))


def main() -> None:
    csv_path = Path("customer_data_raw.csv")

    df = load_and_inspect(csv_path)
    df = drop_duplicates(df)
    df = standardise_city_names(df)
    df = handle_missing_values(df)
    df = remove_outliers(df)
    verify_clean(df)

    df.to_csv("customer_data_cleaned.csv", index=False)
    print("\nSaved cleaned dataset to customer_data_cleaned.csv")


if __name__ == "__main__":
    main()
