"""
W1D4: Exploratory Data Analysis (EDA)
Task: load a real dataset, run summary diagnostics, plot distributions,
correlation heatmap, and top category counts, then write a narrative.

Author: Siriyala Nishar
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def summary_diagnostics(df: pd.DataFrame) -> None:
    """Run describe(), info(), and isnull().sum() — the standard first
    look at any new dataset before doing anything else.
    """
    print("=== df.describe() ===")
    print(df.describe())
    print("\n=== df.info() ===")
    df.info()
    print("\n=== df.isnull().sum() ===")
    print(df.isnull().sum())


def plot_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot the distribution of each numeric column as a histogram grid."""
    numeric_cols = df.select_dtypes(include="number").columns.drop("employee_id")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flat, numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(out_dir / "distributions.png", dpi=120)
    plt.close()
    print(f"Saved distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot a correlation heatmap of numeric columns."""
    numeric_cols = df.select_dtypes(include="number").drop(columns=["employee_id"])
    corr = numeric_cols.corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=120)
    plt.close()
    print(f"Saved correlation_heatmap.png")


def plot_top_categories(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot top-10 category counts for the department column."""
    counts = df["department"].value_counts().head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts.values, y=counts.index)
    plt.title("Top Department Counts")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "top_categories.png", dpi=120)
    plt.close()
    print(f"Saved top_categories.png")


def five_observations(df: pd.DataFrame) -> None:
    """Print 5 concrete observations derived from the diagnostics above."""
    obs = [
        f"1. Dataset has {df.shape[0]} rows and {df.shape[1]} columns.",
        f"2. 'satisfaction_score' has {df['satisfaction_score'].isnull().sum()} "
        f"missing values ({df['satisfaction_score'].isnull().mean():.1%} of rows).",
        f"3. 'monthly_salary' max is {df['monthly_salary'].max():,.0f}, which is "
        f"far above the 75th percentile ({df['monthly_salary'].quantile(0.75):,.0f}) "
        f"— likely a data entry outlier.",
        f"4. 'department' has {df['department'].nunique()} unique categories, "
        f"with '{df['department'].mode()[0]}' being most common.",
        f"5. 'age' and 'experience_years' are likely correlated, since experience "
        f"is typically derived from age minus a starting work age.",
    ]
    print("\n=== 5 Observations ===")
    for o in obs:
        print(o)


def main() -> None:
    csv_path = Path("employee_dataset.csv")
    out_dir = Path(".")

    df = pd.read_csv(csv_path)

    summary_diagnostics(df)
    plot_distributions(df, out_dir)
    plot_correlation_heatmap(df, out_dir)
    plot_top_categories(df, out_dir)
    five_observations(df)


if __name__ == "__main__":
    main()
