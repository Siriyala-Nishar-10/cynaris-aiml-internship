# W4D5 — 1M Capstone: Sentiment Classifier — Deploy & Document

## Objective

Train and compare two classifiers on a binary sentiment classification
task (positive vs negative review text): Logistic Regression as the
baseline, Random Forest as the comparison model. Evaluate with
`classification_report`, a confusion matrix, and an ROC-AUC curve.

## Files

| File                       | Description                                                                       |
| -------------------------- | --------------------------------------------------------------------------------- |
| `sentiment_classifier.py`  | Builds the dataset, trains both models, evaluates, and saves all output artifacts |
| `confusion_matrices.png`   | Side-by-side confusion matrices for both models                                   |
| `roc_curve_comparison.png` | Overlaid ROC curves with AUC for both models                                      |
| `model_comparison.csv`     | Accuracy / precision / recall / F1 / ROC-AUC side by side                         |
| `results_summary.json`     | Same metrics in JSON, for programmatic use (e.g. MLflow logging)                  |

## Dataset

A synthetically generated set of ~1,200 short product-review sentences
(positive/negative templates across 18 product subjects), built entirely
offline so the capstone is reproducible without external downloads.
Two things were deliberately added to keep the task realistic rather than
trivially separable:

1. **Negation examples** (`"not bad"`, `"isn't great but not terrible"`,
   etc.) — TF-IDF has no notion of word order, so a sentiment word next to
   a negator is genuinely hard for a bag-of-words model. This is the kind
   of failure mode worth knowing about before shipping a text classifier.
2. **5% random label noise** — simulates the annotation error you'd expect
   in a real human-labeled dataset, so accuracy lands in a believable range
   instead of a meaningless 100%.

Swap `build_dataset()` for `pd.read_csv("your_reviews.csv")` to run this
against a real dataset (e.g. IMDB reviews, Amazon reviews) — everything
downstream only assumes a `text` column and a binary `label` column.

## Results

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
| ------------------- | -------- | --------- | ------ | ----- | ------- |
| Logistic Regression | 0.943    | 0.935     | 0.954  | 0.944 | 0.944   |
| Random Forest       | 0.933    | 0.917     | 0.954  | 0.935 | 0.940   |

Logistic Regression came out slightly ahead on every metric.

## How to run

```bash
pip install scikit-learn pandas matplotlib
python sentiment_classifier.py
```

## Author

Siriyala Nishar
