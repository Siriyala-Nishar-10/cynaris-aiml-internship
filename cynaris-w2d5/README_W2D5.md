# W2D5 — Week 2 Project: End-to-End Preprocessing Pipeline

## What This Does

Applies everything from Week 2 (EDA, missing data handling, encoding,
scaling) on a Titanic-style dataset, using a single sklearn
`ColumnTransformer` + `Pipeline` to build a reusable, leak-safe
preprocessing pipeline that exports ML-ready features.

## Tech Stack

- Python 3.14
- Pandas, Scikit-learn, Matplotlib

## Setup

```bash
python -m venv venv
source venv/Scripts/activate
pip install pandas scikit-learn matplotlib
pip freeze > requirements.txt
```

## Run

```bash
python week2_pipeline.py
```

## Pipeline Design

**Why `ColumnTransformer` + `Pipeline` instead of manual step-by-step
code (like W2D1/W2D2):** it bundles imputation, encoding, and scaling
into ONE object that is fit only on training data, then applied
(never re-fit) to test data. This prevents **data leakage** — if you
imputed missing values or scaled using statistics from the _full_
dataset (train + test combined) before splitting, information from
the test set would leak into training, giving an unrealistically
optimistic performance estimate.

**Numeric pipeline** (`age`, `fare`, `sibsp`, `parch`):
`SimpleImputer(median)` → `StandardScaler()`

**Categorical pipeline** (`sex`, `embarked`, `pclass`):
`SimpleImputer(most_frequent)` → `OneHotEncoder(drop first)`

## Key EDA Findings

- **Survival by sex:** female 62.2%, male 14.3% — the well-known
  "women and children first" pattern
- **Survival by class:** 1st class 53.1%, 2nd 26.8%, 3rd 24.5%
- **Missing data:** age (17.5%), fare (1.25%), embarked (1.25%) —
  age's missingness is realistic and substantial, handled via median
  imputation in the pipeline

## Output

- `eda_survival_rates.png` — survival rate by class and sex
- `titanic_train_ml_ready.csv` / `titanic_test_ml_ready.csv` — fully
  numeric, encoded, scaled, ML-ready features split before any
  transformation was fit (leak-safe)

## Dataset

`titanic_like.csv` — 400 synthetic passenger records modeled on real
Titanic survival patterns (class, sex, age effects), with realistic
missingness injected in `age` and `fare`.

## Author

Siriyala Nishar
