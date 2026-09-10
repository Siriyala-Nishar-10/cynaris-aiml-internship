"""
W2D1: Feature Engineering & Encoding
Task: apply categorical encoders, feature scalers, and SelectKBest
feature selection on a loan applicant dataset.

Author: Siriyala Nishar
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    StandardScaler, MinMaxScaler, RobustScaler,
)
from sklearn.feature_selection import SelectKBest, f_classif


def apply_encoders(df: pd.DataFrame) -> pd.DataFrame:
    """Apply LabelEncoder, OneHotEncoder, and OrdinalEncoder, noting
    the trade-off of each on the categorical columns.

    Trade-offs:
    - LabelEncoder: assigns arbitrary integers (0,1,2...) to categories.
      Fine for the TARGET column, but risky for FEATURES since it
      implies a false ordinal relationship (e.g. 'Tier 3' > 'Tier 1'
      numerically, which the model may wrongly treat as meaningful).
    - OneHotEncoder: creates a binary column per category, no false
      ordering — correct choice for nominal (unordered) categories
      like 'employment_type'. Downside: increases dimensionality.
    - OrdinalEncoder: like LabelEncoder but lets you SPECIFY the true
      order — correct choice for genuinely ordered categories like
      'education' (High School < Bachelors < Masters < PhD).
    """
    # LabelEncoder on the target (appropriate use — target, not a feature)
    le = LabelEncoder()
    df["loan_approved_encoded"] = le.fit_transform(df["loan_approved"])

    # OneHotEncoder on nominal categorical features (no inherent order)
    ohe = OneHotEncoder(sparse_output=False, drop="first")
    ohe_cols = ohe.fit_transform(df[["employment_type", "city_tier"]])
    ohe_df = pd.DataFrame(
        ohe_cols, columns=ohe.get_feature_names_out(["employment_type", "city_tier"])
    )

    # OrdinalEncoder on genuinely ordered categorical feature
    education_order = [["High School", "Bachelors", "Masters", "PhD"]]
    oe = OrdinalEncoder(categories=education_order)
    df["education_encoded"] = oe.fit_transform(df[["education"]])

    result = pd.concat([df.reset_index(drop=True), ohe_df], axis=1)
    print("Encoded columns added:", list(ohe_df.columns) + ["education_encoded"])
    return result


def apply_scalers(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Apply StandardScaler, MinMaxScaler, RobustScaler on numeric
    columns and plot before/after distributions for comparison.
    """
    numeric_col = "annual_income"  # has some spread, good demo column
    values = df[[numeric_col]].values

    standard = StandardScaler().fit_transform(values)
    minmax = MinMaxScaler().fit_transform(values)
    robust = RobustScaler().fit_transform(values)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].hist(values, bins=30, color="steelblue")
    axes[0].set_title("Original")
    axes[1].hist(standard, bins=30, color="orange")
    axes[1].set_title("StandardScaler")
    axes[2].hist(minmax, bins=30, color="green")
    axes[2].set_title("MinMaxScaler")
    axes[3].hist(robust, bins=30, color="purple")
    axes[3].set_title("RobustScaler")
    plt.tight_layout()
    plt.savefig(out_dir / "scaling_comparison.png", dpi=120)
    plt.close()
    print("Saved scaling_comparison.png")

    df["income_standard_scaled"] = standard
    df["income_minmax_scaled"] = minmax
    df["income_robust_scaled"] = robust
    return df


def select_top_features(df: pd.DataFrame) -> list[str]:
    """Use SelectKBest to identify the top 5 features predicting
    loan approval, and document why each plausibly matters.
    """
    feature_cols = [
        "age", "annual_income", "credit_score", "loan_amount",
        "years_at_job", "education_encoded",
        "employment_type_Self-Employed", "employment_type_Unemployed",
        "city_tier_Tier 2", "city_tier_Tier 3",
    ]
    X = df[feature_cols].fillna(0)
    y = df["loan_approved"]

    selector = SelectKBest(score_func=f_classif, k=5)
    selector.fit(X, y)

    scores = pd.Series(selector.scores_, index=feature_cols).sort_values(ascending=False)
    top5 = scores.head(5)
    print("\nTop 5 features by SelectKBest (F-score):")
    print(top5)

    reasons = {
        "credit_score": "Directly reflects creditworthiness — the strongest standard loan-approval signal.",
        "annual_income": "Higher income indicates greater repayment capacity.",
        "loan_amount": "Larger requested amounts carry more risk relative to income.",
        "years_at_job": "Job stability is a common proxy for reliable future income.",
        "age": "Correlates with financial stability and credit history length.",
        "education_encoded": "Higher education often correlates with earning potential.",
    }
    print("\nWhy these matter:")
    for feat in top5.index:
        print(f"- {feat}: {reasons.get(feat, 'Relevant to repayment capacity/risk profile.')}")

    return list(top5.index)


def main() -> None:
    csv_path = Path("loan_applicants.csv")
    out_dir = Path(".")

    df = pd.read_csv(csv_path)
    print("Loaded shape:", df.shape)

    df = apply_encoders(df)
    df = apply_scalers(df, out_dir)
    top_features = select_top_features(df)

    df.to_csv("loan_applicants_engineered.csv", index=False)
    print(f"\nSaved engineered dataset. Top 5 features: {top_features}")


if __name__ == "__main__":
    main()
