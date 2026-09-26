# W4D1: Model Evaluation — K-Fold CV, Stratified CV, Learning Curves (+ Precision/Recall/AUC)

## Objective

Understand why a single train/test split isn't a reliable evaluation,
compare `KFold` vs `StratifiedKFold`, and use learning curves to
diagnose overfitting vs underfitting — plus round out the evaluation
with precision/recall/F1/ROC-AUC.

## Note on this lesson's page

The page title/video said "Precision, Recall, AUC," but the tagline,
all 3 resource links, and the pre-loaded Code Playground starter code
(`learning_curve`, `StratifiedKFold`, multiple classifiers) all pointed
to k-fold CV and learning curves instead. Built for the CV/learning
curves content since the playground starter code was the strongest,
most concrete signal of the intended task — and folded in
precision/recall/F1/ROC-AUC as supporting metrics so the title isn't
ignored either.

## What's in this folder

- `w4d1_cv_learning_curves.py` — main script
- `learning_curves.png` — train vs validation score for 3 models
- `output_evidence.txt` — captured console output

## Approach

### 1. Why one split isn't enough

Ran the _same_ Logistic Regression model on 10 different random
train/test splits of the Breast Cancer dataset:

- **Accuracy ranged from 0.9649 to 1.0000** (a 0.0351 spread) depending
  purely on which split you happened to get.
- This is the concrete case for cross-validation: any single split's
  score is one noisy sample, not a reliable estimate.

### 2. KFold vs StratifiedKFold

| Method          | Mean   | Std    |
| --------------- | ------ | ------ |
| KFold           | 0.9771 | 0.0090 |
| StratifiedKFold | 0.9737 | 0.0166 |

Interesting result: on this run, plain `KFold` actually had _lower_
variance than `StratifiedKFold`. This is because the Breast Cancer
dataset is only mildly imbalanced (37%/63%), and `shuffle=True` let
plain KFold land on well-balanced folds by chance. In theory,
StratifiedKFold guarantees every fold matches the overall class
balance, which matters far more on smaller or more heavily imbalanced
datasets (e.g. W2D3's fraud dataset), where an unlucky KFold split
could leave a fold with very few or zero minority-class examples.
Documented the actual result rather than asserting a textbook claim
the data didn't support.

### 3. Precision / Recall / F1 / ROC-AUC via 5-fold Stratified CV

| Model               | Precision | Recall | F1     | ROC-AUC |
| ------------------- | --------- | ------ | ------ | ------- |
| Logistic Regression | 0.9683    | 0.9916 | 0.9794 | 0.9953  |
| Decision Tree       | 0.9320    | 0.9581 | 0.9438 | 0.9068  |
| Random Forest       | 0.9624    | 0.9665 | 0.9637 | 0.9895  |

Each metric is the mean across 5 stratified folds, not a single-split
number — consistent with point 1's lesson.

### 4. Learning curves — diagnosing overfitting vs underfitting

| Model                        | Final Train | Final Val | Gap    | Diagnosis   |
| ---------------------------- | ----------- | --------- | ------ | ----------- |
| Decision Tree (max_depth=5)  | 0.9921      | 0.9280    | 0.0641 | Overfitting |
| Decision Tree (unrestricted) | 1.0000      | 0.9104    | 0.0896 | Overfitting |
| Random Forest                | 1.0000      | 0.9543    | 0.0457 | Overfitting |

All three show train accuracy pinned near 1.0 with validation accuracy
trailing behind — the classic overfitting signature (high variance).
The unrestricted tree has the widest gap, as expected; Random Forest
has the narrowest, showing how averaging many trees reduces (but here,
doesn't fully eliminate) the overfitting a single tree exhibits.

## Key concepts

- **Why not a single split**: one split's score is a single noisy
  sample of model performance, not a reliable estimate.
- **K-Fold CV**: averages performance over k different splits for a
  more stable estimate.
- **Stratified K-Fold**: preserves class balance in every fold —
  matters more as class imbalance and dataset size get more extreme.
- **Learning curves**: plot train vs validation score as training set
  size grows.
  - Persistent gap, high train score → **overfitting** (high variance)
  - Both scores low and converged → **underfitting** (high bias)
  - Both high and close together → good fit

## Self-review checklist

- [x] Code runs end-to-end with no errors
- [x] Comments explain _why_, not just _what_
- [x] Demonstrated split-to-split variance concretely (10 splits)
- [x] Compared KFold vs StratifiedKFold and reported the actual result
      honestly, even though it didn't match the textbook expectation
- [x] Metrics computed via CV, not a single split
- [x] Learning curves plotted and diagnosed for 3 models
- [x] Reproducible (`random_state` fixed throughout)
- [x] Output evidence + plot captured
- [x] Committed with descriptive messages, pushed to feature branch

## Author

Siriyala Nishar
