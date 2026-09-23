# W2D2 — Missing Data Handling: Imputation Strategies

## What This Does

Compares 3 imputation strategies — SimpleImputer, KNNImputer, and
IterativeImputer (MICE-style) — on a dataset with 3 different types of
missingness (random, correlated with a category, and a large missing
block), documenting when each strategy is appropriate.

## Tech Stack

- Python 3.14
- Pandas, NumPy, Scikit-learn

## Setup

```bash
python -m venv venv
source venv/Scripts/activate
pip install pandas numpy scikit-learn
pip freeze > requirements.txt
```

## Run

```bash
python missing_data_handling.py
```

## Strategy Comparison

| Strategy                    | How it works                                                                                          | When to use                                                               | Trade-off                                          |
| --------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------- |
| **SimpleImputer**           | Fills with a single statistic (mean/median/mode) for the whole column                                 | Fast baseline, low missingness, no relationship needed with other columns | Ignores relationships between features — naive     |
| **KNNImputer**              | Fills using the average of the k most similar rows (by other numeric features)                        | Missing values likely relate to other numeric features                    | Slower, needs numeric/scaled input, sensitive to k |
| **IterativeImputer (MICE)** | Models each feature with missing values as a function of all other features, iterating to convergence | Complex multivariate relationships between features                       | Slowest, can be unstable on small/noisy data       |

## Missingness Patterns in the Dataset

- `age` — missing completely at random (MCAR), 7.5%
- `department` — missing completely at random (MCAR), 5%
- `annual_income` — missing at random (MAR), concentrated in the Sales
  department (10%) — simulates a real scenario where a specific group
  under-reports a field
- `years_experience` — a large contiguous missing block (15%),
  simulating a bad data collection batch — the most informative case
  for comparing imputation strategies, since simple mean/median would
  be a poor fit for 30 consecutive missing rows

## Dataset

`employee_missing_data.csv` — 200 synthetic employee records with
deliberately varied missingness patterns to demonstrate when a simple
strategy is sufficient vs. when a smarter one is needed.

## Author

Siriyala Nishar
