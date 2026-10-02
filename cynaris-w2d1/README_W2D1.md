# W2D1 — Feature Engineering & Encoding

## What This Does

Applies categorical encoding (LabelEncoder, OneHotEncoder,
OrdinalEncoder), numeric scaling (StandardScaler, MinMaxScaler,
RobustScaler), and SelectKBest feature selection on a loan applicant
dataset, documenting the reasoning behind each choice.

## Tech Stack

- Python 3.14
- Pandas, NumPy, Scikit-learn, Matplotlib

## Setup

```bash
python -m venv venv
source venv/Scripts/activate
pip install pandas numpy scikit-learn matplotlib
pip freeze > requirements.txt
```

## Run

```bash
python feature_engineering.py
```

## Encoding Trade-offs

- **LabelEncoder** — assigns arbitrary integers to categories. Used
  correctly here only on the target column; risky on features since
  it implies a false ordinal relationship.
- **OneHotEncoder** — creates a binary column per category, no false
  ordering. Correct choice for nominal (unordered) categories like
  `employment_type` and `city_tier`. Trade-off: increases dimensionality.
- **OrdinalEncoder** — like LabelEncoder but lets you specify the true
  order. Correct choice for genuinely ordered categories like
  `education` (High School < Bachelors < Masters < PhD).

## Scaling Comparison

`StandardScaler` (mean 0, std 1) is sensitive to outliers since it
uses mean/std. `MinMaxScaler` compresses to a fixed [0,1] range but is
even more outlier-sensitive (one extreme value distorts the whole
scale). `RobustScaler` uses median/IQR instead, making it the most
outlier-resistant of the three — see `scaling_comparison.png` for the
visual before/after comparison on `annual_income`.

## Top 5 Features (SelectKBest, F-score)

1. `credit_score` — strongest standard loan-approval signal
2. `employment_type_Self-Employed` — affects repayment risk profile
3. `annual_income` — reflects repayment capacity
4. `age` — proxy for financial stability
5. `city_tier_Tier 3` — regional risk/cost-of-living factor

## Dataset

`loan_applicants.csv` — 250 synthetic loan applications with mixed
categorical (education, city tier, employment type) and numeric
(age, income, credit score, loan amount) features, and a binary
approval target correlated with credit score and income.

## Author

Siriyala Nishar
