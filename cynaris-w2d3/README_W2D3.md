# W2D3 — Handling Imbalanced Data: SMOTE

## What This Does

Demonstrates why class imbalance breaks naive accuracy-based
evaluation, applies SMOTE (Synthetic Minority Oversampling Technique)
to rebalance the training data, and shows the real before/after
trade-off in model performance on a fraud detection dataset.

## Tech Stack

- Python 3.14
- Pandas, Scikit-learn, imbalanced-learn (SMOTE), Matplotlib

## Setup

```bash
python -m venv venv
source venv/Scripts/activate
pip install pandas scikit-learn imbalanced-learn matplotlib
pip freeze > requirements.txt
```

## Run

```bash
python smote_imbalanced.py
```

## Results

| Metric    | No SMOTE  | With SMOTE |
| --------- | --------- | ---------- |
| Accuracy  | 0.956     | 0.800      |
| Precision | 0.667     | 0.172      |
| Recall    | **0.167** | **0.833**  |
| F1        | 0.267     | 0.286      |

## The Key Insight

Without SMOTE, the model achieves **95.6% accuracy** — but this is
misleading: it only catches **2 of 12 actual fraud cases** (16.7%
recall), because with 95% of the data being legitimate transactions,
a model can score well on accuracy just by predicting "not fraud"
almost every time.

After SMOTE rebalances the _training_ set (not the test set — SMOTE
is applied only to training data to avoid leaking synthetic patterns
into evaluation), the model catches **10 of 12 fraud cases** (83.3%
recall), at the cost of more false positives (lower precision, lower
raw accuracy).

**For fraud detection specifically, this trade-off is usually correct**
— missing real fraud is far costlier than investigating a false alarm,
so prioritising recall over raw accuracy is the right business decision.

## What Each Function Does

- `inspect_class_balance()` — makes the 95/5 imbalance explicit before
  doing anything else.
- `train_baseline_no_smote()` — trains without addressing imbalance,
  to show why accuracy alone is misleading here.
- `apply_smote_and_retrain()` — applies SMOTE only to the training
  set, retrains, and reports the same metrics for comparison.
- `plot_comparison()` — bar chart comparing all 4 metrics before/after.

## Dataset

`transaction_fraud.csv` — 1000 synthetic transactions (950 legitimate,
50 fraudulent — a realistic 5% fraud rate), with deliberately
overlapping feature distributions between classes so the classification
task is genuinely hard, not trivially separable.

## Author

Siriyala Nishar
