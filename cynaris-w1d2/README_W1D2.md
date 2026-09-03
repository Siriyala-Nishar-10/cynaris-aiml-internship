# W1D2 — Pandas for Data Manipulation

## What This Does

Loads an Indian retail orders dataset and demonstrates core Pandas
operations: filtering, groupby aggregation, merging with metadata,
and pivot tables — then exports the cleaned result to both CSV and
Parquet, comparing file sizes.

## Tech Stack

- Python 3.14
- Pandas 3.0.5
- PyArrow (for Parquet export)

## Setup

```bash
python -m venv venv
source venv/Scripts/activate
pip install pandas pyarrow
pip freeze > requirements.txt
```

## Run

```bash
python pandas_manipulation.py
```

## What Each Function Does

- `load_and_inspect(csv_path)` — loads the CSV, prints shape, dtypes, and
  the first 10 rows to understand the data before doing anything else.
- `filter_orders(df)` — filters rows using a boolean mask
  (`df[df["total_amount"] > 5000]`), keeping only high-value orders.
- `groupby_city_category(df)` — groups by two columns at once (`city`,
  `category`) and sums revenue per group, then sorts to find top combos.
- `merge_with_city_metadata(df)` — performs a `left` merge to attach a
  small city→region lookup table onto every order row, without losing
  any original rows.
- `pivot_revenue_by_region_category(df)` — reshapes the data into a
  region x category revenue matrix using `pivot_table`, useful for
  quick cross-tab reporting.
- `export_and_compare(df, base_name)` — writes the final DataFrame to
  both `.csv` and `.parquet`, then compares file sizes on disk.

## Sample Output (Key Results)

```
Shape: (200, 7)
Filtered 104 of 200 orders with total > 5000
Top city-category revenue combo: Hyderabad / Apparel — ₹91,397.36
CSV size: 11,571 bytes
Parquet size: 10,473 bytes
Parquet is 1.10x smaller than CSV
```

## Dataset

`indian_retail_orders.csv` — 200 synthetic e-commerce orders across 6
Indian cities (Bangalore, Mumbai, Delhi, Chennai, Hyderabad, Pune) and
5 product categories, generated to resemble a realistic retail dataset
for practicing filter/groupby/merge/pivot operations.

## Author

Siriyala Nishar
