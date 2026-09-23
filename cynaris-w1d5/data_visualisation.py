"""
W1D5: Data Visualisation — Matplotlib & Seaborn
Task: build a range of visualization types (beyond basic histograms)
to compare, correlate, and reveal trends in a real dataset.

Author: Siriyala Nishar
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def boxplot_salary_by_department(df: pd.DataFrame, out_dir: Path) -> None:
    """Boxplot: salary spread per department.

    Why: a boxplot shows median, quartiles, and outliers per group at
    once — better than a bar chart of means, which would hide the
    salary outlier and the actual spread within each department.
    """
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="department", y="monthly_salary")
    plt.xticks(rotation=30)
    plt.title("Monthly Salary Distribution by Department")
    plt.tight_layout()
    plt.savefig(out_dir / "salary_by_department_boxplot.png", dpi=120)
    plt.close()
    print("Saved salary_by_department_boxplot.png")


def scatter_experience_vs_salary(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter plot: experience vs salary, coloured by department.

    Why: a scatter plot is the right choice for showing the
    relationship between two continuous variables, and colour-coding
    by department reveals whether the relationship differs by group.
    """
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df, x="experience_years", y="monthly_salary",
        hue="department", alpha=0.7
    )
    plt.title("Experience vs Monthly Salary by Department")
    plt.tight_layout()
    plt.savefig(out_dir / "experience_vs_salary_scatter.png", dpi=120)
    plt.close()
    print("Saved experience_vs_salary_scatter.png")


def violin_satisfaction_by_department(df: pd.DataFrame, out_dir: Path) -> None:
    """Violin plot: satisfaction score distribution shape per department.

    Why: a violin plot shows the full distribution shape (not just
    quartiles like a boxplot), useful for spotting bimodal patterns
    e.g. some departments having two distinct groups of satisfaction.
    """
    plt.figure(figsize=(9, 6))
    sns.violinplot(data=df, x="department", y="satisfaction_score")
    plt.xticks(rotation=30)
    plt.title("Satisfaction Score Distribution by Department")
    plt.tight_layout()
    plt.savefig(out_dir / "satisfaction_violin.png", dpi=120)
    plt.close()
    print("Saved satisfaction_violin.png")


def bar_average_salary_by_city(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: average salary per city, sorted descending.

    Why: a sorted bar chart is the clearest way to rank categories —
    sorting (rather than leaving alphabetical) makes the highest and
    lowest immediately visible without the reader having to scan.
    """
    avg_salary = (
        df.groupby("city")["monthly_salary"].mean().sort_values(ascending=False)
    )
    plt.figure(figsize=(9, 6))
    sns.barplot(x=avg_salary.values, y=avg_salary.index, hue=avg_salary.index,
                palette="viridis", legend=False)
    plt.title("Average Monthly Salary by City")
    plt.xlabel("Average Monthly Salary")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_salary_by_city_bar.png", dpi=120)
    plt.close()
    print("Saved avg_salary_by_city_bar.png")


def pairplot_numeric_relationships(df: pd.DataFrame, out_dir: Path) -> None:
    """Pairplot: all pairwise relationships between numeric columns.

    Why: a pairplot gives a fast overview of every numeric
    relationship at once — a quicker way to spot patterns worth
    investigating further than making each scatter plot individually.
    """
    numeric_cols = ["age", "experience_years", "monthly_salary", "satisfaction_score"]
    g = sns.pairplot(df[numeric_cols].dropna(), diag_kind="kde")
    g.fig.suptitle("Pairwise Relationships", y=1.02)
    g.savefig(out_dir / "pairplot.png", dpi=120)
    plt.close()
    print("Saved pairplot.png")


def main() -> None:
    csv_path = Path("employee_dataset.csv")
    out_dir = Path(".")

    df = pd.read_csv(csv_path)
    # exclude the known salary outlier for cleaner comparative plots
    df_clean = df[df["monthly_salary"] < 200000].copy()

    boxplot_salary_by_department(df_clean, out_dir)
    scatter_experience_vs_salary(df_clean, out_dir)
    violin_satisfaction_by_department(df, out_dir)
    bar_average_salary_by_city(df_clean, out_dir)
    pairplot_numeric_relationships(df_clean, out_dir)

    print("\nAll visualisations saved successfully.")


if __name__ == "__main__":
    main()
