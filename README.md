# W1D1 — Python for ML: NumPy Fundamentals

## What This Does

This script demonstrates core NumPy operations required for ML work:
array creation across dimensions, broadcasting, vectorised operations,
matrix multiplication, and descriptive statistics computed on a real
dataset (`student_performance.csv`).

## Tech Stack

- Python 3.14
- NumPy 2.5.2
- Pandas 3.0.5

## Setup

```bash
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash
pip install -r requirements.txt
```

## Run

```bash
python numpy_fundamentals.py
```

## What Each Function Does

- `create_arrays()` — builds a 1D, 2D, and 3D NumPy array and prints their `.shape`.
- `broadcasting_and_vectorised_ops(arr_2d)` — adds a smaller array to a larger
  one using broadcasting, and squares every element with a vectorised op —
  no explicit Python loops used.
- `matrix_multiplication(arr_2d)` — performs real linear-algebra matrix
  multiplication with `@`, as opposed to element-wise `*`.
- `dataset_statistics(csv_path)` — loads the CSV with Pandas, converts
  columns to NumPy arrays, and computes mean, standard deviation, and
  Pearson correlation between `study_hours` and `exam_score`.

## Sample Output

```
1D shape: (5,)
2D shape: (2, 3)
3D shape: (2, 2, 2)
Broadcasted result:
 [[11 22 33]
 [14 25 36]]
Matrix multiplication result:
 [[22 28]
 [49 64]]
study_hours_mean: 5.016
study_hours_std: 2.578
exam_score_mean: 53.168
exam_score_std: 19.584
study_vs_score_correlation: 0.966
```

## Dataset

`student_performance.csv` — 50 synthetic student records (`study_hours`,
`sleep_hours`, `exam_score`), generated to have a realistic positive
correlation between study hours and exam performance, for demonstrating
`np.corrcoef`.
