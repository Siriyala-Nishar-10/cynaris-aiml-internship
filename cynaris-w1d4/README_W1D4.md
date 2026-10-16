# W1D4 — Exploratory Data Analysis (EDA)

## What This Does
Runs a full first-look EDA on a synthetic employee dataset: summary
diagnostics (`describe()`, `info()`, missing value counts), distribution
plots for every numeric column, a correlation heatmap, and top-category
counts — then documents 5 concrete observations and a written narrative.

## Tech Stack
- Python 3.14
- Pandas, Matplotlib, Seaborn

## Setup
```bash
python -m venv venv
source venv/Scripts/activate
pip install pandas matplotlib seaborn
pip freeze > requirements.txt
```

## Run
```bash
python eda_analysis.py
```

## What Each Function Does
- `summary_diagnostics(df)` — runs `describe()`, `info()`, and
  `isnull().sum()`, the standard first three checks on any new dataset.
- `plot_distributions(df, out_dir)` — histogram + KDE for every numeric
  column, saved as `distributions.png`.
- `plot_correlation_heatmap(df, out_dir)` — correlation matrix of
  numeric columns as a heatmap, saved as `correlation_heatmap.png`.
- `plot_top_categories(df, out_dir)` — bar chart of department counts,
  saved as `top_categories.png`.
- `five_observations(df)` — prints 5 concrete, data-derived observations
  (missing values, outliers, category distribution, expected
  correlations).

## Output Files
- `distributions.png`
- `correlation_heatmap.png`
- `top_categories.png`
- `EDA_NARRATIVE.md` — 200+ word written narrative on findings

## Dataset
`employee_dataset.csv` — 300 synthetic employee records (age,
department, city, experience, salary, satisfaction score), with
deliberately injected missing values and one salary outlier, to
practice realistic EDA.

## AI Usage Note
Built with AI assistance (Claude) for code structure, plot selection,
and narrative drafting. Reviewed and tested locally — verified plots
render correctly and observations match the printed diagnostics.

## Author
Siriyala Nishar
