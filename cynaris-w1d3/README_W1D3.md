# W1D3 — Data Loading, Cleaning & Inspection

## What This Does
Loads a raw, messy customer dataset and cleans it: removes duplicate
rows, standardises inconsistent text formatting, imputes missing
values, and removes implausible outliers — then verifies and saves
the cleaned result.

## Tech Stack
- Python 3.14
- Pandas, NumPy

## Setup
```bash
python -m venv venv
source venv/Scripts/activate
pip install pandas numpy
pip freeze > requirements.txt
```

## Run
```bash
python data_cleaning.py
```

## What Each Function Does
- `load_and_inspect(csv_path)` — loads the CSV and reports shape,
  dtypes, missing value counts, and duplicate count before touching
  anything, so cleaning decisions are based on evidence, not guesses.
- `drop_duplicates(df)` — removes exact duplicate rows.
- `standardise_city_names(df)` — fixes inconsistent casing (e.g.
  "bangalore" vs "Bangalore") using `.str.title()`, since inconsistent
  casing silently breaks groupby/filter operations.
- `handle_missing_values(df)` — imputes missing numeric columns (age,
  income) with the column median (robust to outliers), and drops rows
  with missing city (a categorical field with no reasonable default).
- `remove_outliers(df)` — removes rows with an implausible age (>100),
  based on domain knowledge rather than a purely statistical rule.
- `verify_clean(df)` — confirms zero missing values, zero duplicates,
  and a sane age range after cleaning.

## Results
| Metric | Before | After |
|---|---|---|
| Rows | 123 | 101 |
| Missing values | 39 total | 0 |
| Duplicate rows | 3 | 0 |
| City casing variants | 2 (Bangalore/bangalore) | 1 (standardised) |
| Max age | 150 (data error) | 64 |

## Dataset
`customer_data_raw.csv` — 120 synthetic customer records with
deliberately injected data quality issues (missing values, duplicate
rows, inconsistent city casing, one unrealistic age value) to practice
realistic data cleaning.

## AI Usage Note
Built with AI assistance (Claude) for code structure and to generate
a realistic messy dataset for practice. Reviewed and tested locally —
verified cleaning logic against the printed before/after summary
(row counts, missing values, duplicate counts) to confirm correctness.

## Author
Siriyala Nishar
