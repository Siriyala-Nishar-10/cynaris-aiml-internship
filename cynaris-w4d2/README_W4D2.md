# W4D2: Bias-Variance Tradeoff & Regularisation (L1 vs L2)

## Objective

Understand the bias-variance tradeoff concretely via polynomial model
complexity, and how L1 (Lasso) vs L2 (Ridge) regularization each
control that tradeoff — with Lasso additionally performing automatic
feature selection.

## What's in this folder

- `w4d2_bias_variance_regularization.py` — main script
- `bias_variance_curve.png` — train/test MSE vs polynomial degree
- `underfit_good_overfit_examples.png` — 3 fitted curves against the data
- `l1_vs_l2_coefficients.png` — coefficient comparison across models
- `ridge_alpha_sweep.png` — regularization strength vs bias-variance
- `output_evidence.txt` — captured console output

## Approach

### 1. Bias-variance tradeoff via polynomial degree

Fit polynomials of increasing degree (1 to 15) to noisy cubic data and
tracked train vs test MSE at each:

| Degree | Train MSE | Test MSE | Regime       |
| ------ | --------- | -------- | ------------ |
| 1      | 34.29     | 54.89    | Underfitting |
| 2      | 10.62     | 13.41    | Underfitting |
| 3      | 7.74      | 9.99     | Near-optimal |
| 4      | 7.74      | 10.10    | Near-optimal |
| 6      | 7.53      | 10.31    | Near-optimal |
| 9      | 7.51      | 10.32    | Overfitting  |
| 12     | 7.34      | 11.19    | Overfitting  |
| 15     | 7.12      | 36.74    | Overfitting  |

Classic U-shape: too-simple models have high error on _both_ train and
test (high bias); too-complex models get train error near zero but
test error blows up (high variance) — degree 15's test MSE (36.74)
nearly matches degree 1's (54.89), despite fitting training data far
better.

### 2. L1 (Lasso) vs L2 (Ridge) on irrelevant features

Built a dataset with 20 features, only 5 of which actually matter.

| Model                          | Test MSE | Coefficients ≈ 0 |
| ------------------------------ | -------- | ---------------- |
| Linear Regression (no penalty) | 5.826    | 0/20             |
| Ridge (L2, alpha=1.0)          | 5.629    | 1/20             |
| Lasso (L1, alpha=0.1)          | 4.930    | 10/20            |

Lasso zeroed out **10 of 20** coefficients (including most of the 15
genuinely irrelevant features), performing automatic feature selection.
Ridge shrank coefficients but almost never hit exactly zero — it keeps
every feature, just with reduced influence. This is the core structural
difference between L1 and L2 penalties, not just "more/less shrinkage."

### 3. Regularization strength (alpha) vs bias-variance

Swept Ridge's `alpha` on a harder dataset (40 samples, 30 features, 4
informative) to show the same U-shape through a regularization
hyperparameter instead of model complexity:

- **Best alpha:** 2.336 (test MSE = 65.08)
- **Near-unregularized (alpha=0.01):** test MSE = 144.54 (overfitting —
  too many noisy coefficients fit to training noise)
- **Heavily regularized (alpha=1000):** test MSE = 83.02 (underfitting —
  coefficients shrunk too close to zero, losing real signal)

Note: this required deliberately constructing a scenario where
unregularized OLS actually overfits (more features than a comfortable
margin over samples) — an easier dataset showed regularization not
helping at all, which wouldn't have demonstrated the tradeoff.

## Key concepts

- **Bias**: error from an overly simple model that can't capture the
  true pattern (underfitting).
- **Variance**: error from a model too sensitive to the specific
  training sample, capturing noise as if it were signal (overfitting).
- **L2 (Ridge)** penalty: adds `alpha * sum(coef^2)` to the loss.
  Shrinks all coefficients smoothly toward zero, rarely exactly zero.
- **L1 (Lasso)** penalty: adds `alpha * sum(|coef|)` to the loss. Can
  drive coefficients exactly to zero — effectively performing feature
  selection.
- **Regularization strength (alpha)**: alpha=0 is unregularized (can
  overfit); alpha too high shrinks everything toward zero (underfits);
  the right alpha sits at the minimum of the test-error U-curve.

## Self-review checklist

- [x] Code runs end-to-end with no errors
- [x] Comments explain _why_, not just _what_
- [x] Bias-variance tradeoff shown via both model complexity (degree)
      and regularization strength (alpha) — two angles on one concept
- [x] L1 vs L2 difference shown structurally (sparsity), not just via MSE
- [x] Iterated on the alpha-sweep dataset until it actually showed a
      U-shape, rather than reporting a flat/uninteresting result
- [x] Reproducible (`RandomState(42)` throughout)
- [x] Output evidence + 4 plots captured
- [x] Committed with descriptive messages, pushed to feature branch

## Author

Siriyala Nishar
