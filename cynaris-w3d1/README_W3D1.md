# W3D1 — Linear Regression: Scikit-Learn

## What This Does

Trains LinearRegression, Ridge, and Lasso on a house price dataset,
evaluates each with MSE/RMSE/MAE/R², plots predicted-vs-actual and
residuals, and compares all 3 in a results table.

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
python linear_regression.py
```

## Results

| Model             | MSE         | RMSE   | MAE    | R²     |
| ----------------- | ----------- | ------ | ------ | ------ |
| Linear Regression | 461,662,768 | 21,486 | 18,047 | 0.9283 |
| Ridge (α=1.0)     | 460,960,681 | 21,470 | 17,995 | 0.9285 |
| Lasso (α=1.0)     | 461,661,273 | 21,486 | 18,047 | 0.9283 |

**R² of 0.928 means the model explains 92.8% of the variance in house
prices** — a strong fit, expected since the synthetic data was built
from a genuinely linear price formula (sqft, bedrooms, bathrooms, age,
distance to city, garage) plus noise.

All 3 models perform nearly identically here — expected, since with
only 6 features and no severe multicollinearity, there's little for
Ridge/Lasso's regularization to meaningfully improve over plain OLS.
Regularization's benefit shows up more with many/correlated features
or smaller datasets prone to overfitting — neither applies strongly here.

## Why Scale Features for Ridge/Lasso

Features were standardized (`StandardScaler`) before fitting Ridge and
Lasso, since their penalty terms are scale-sensitive — a feature with
a larger numeric range (like `sqft`, in the thousands) would otherwise
be penalized differently than one with a smaller range (like
`bedrooms`, 1-5), purely due to units rather than true importance.

## Output Files

- `predicted_vs_actual.png` — scatter plots for all 3 models vs. the
  perfect-prediction line
- `residuals.png` — residual plots (should show no obvious pattern
  for a well-fit linear model — random scatter around zero)
- `model_comparison.csv` — the results table

## Dataset

`house_prices.csv` — 300 synthetic house records built from a known
linear price formula (sqft, bedrooms, bathrooms, age, distance to
city, garage) plus Gaussian noise, so the true relationship is
genuinely linear — appropriate for demonstrating linear regression.

## Author

Siriyala Nishar
