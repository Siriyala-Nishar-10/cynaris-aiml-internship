# W3D2: Logistic Regression & Classification

## Objective

Learn the sigmoid function, decision boundaries, and multi-class
classification using `LogisticRegression`, and evaluate with proper
classification metrics (not regression metrics).

## What's in this folder

- `w3d2_logistic_regression.py` — main script
- `sigmoid_function.png` — sigmoid curve, sigma(z) = 1/(1+e^-z)
- `roc_curve.png` — ROC curve for the binary classifier
- `decision_boundary.png` — visualized decision boundaries for 3-class Iris
- `output_evidence.txt` — captured console output

## Approach

### 1. Sigmoid function

Plotted `sigma(z) = 1 / (1 + e^-z)` to show how any real-valued linear
combination `z = w·x + b` gets squashed into a probability in (0, 1).
This is what separates Logistic Regression from Linear Regression: the
output is a probability, not a raw numeric prediction.

### 2. Binary classification — Breast Cancer dataset

- Stratified train/test split, features scaled with `StandardScaler`
  (LogisticRegression is gradient-based, so scaling matters — see W2D4).
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix,
  full classification report.
- Printed top-5 feature coefficients by absolute value to show which
  features most strongly push the prediction toward malignant/benign.

### 3. Decision boundaries + multi-class — Iris dataset

- Used only 2 features (sepal length, sepal width) so the decision
  boundary is visualizable in 2D.
- `LogisticRegression` handles 3 classes natively via the multinomial
  (softmax) generalization of the binary sigmoid — no need for manual
  one-vs-rest wrapping.
- Plotted the resulting decision regions as a filled contour with the
  actual data points overlaid.

## Results

**Binary classification (Breast Cancer):**
| Metric | Score |
|-----------|--------|
| Accuracy | 0.9825 |
| Precision | 0.9861 |
| Recall | 0.9861 |
| F1 Score | 0.9861 |
| ROC-AUC | 0.9954 |

**Multi-class (Iris, 2 features only):**
| Metric | Score |
|----------|--------|
| Accuracy | 0.7333 |

(Lower here because only 2 of 4 available features were used, deliberately,
to make the decision boundary plottable in 2D.)

## Key concepts

- **Sigmoid** turns a linear score into a probability; the 0.5 threshold
  on that probability is what creates the (linear) decision boundary.
- **Decision boundary**: for logistic regression this is always linear
  in the feature space (or the transformed feature space, if you add
  polynomial features).
- **Multi-class**: scikit-learn's default `LogisticRegression` uses the
  multinomial (softmax) approach for 3+ classes rather than training
  separate one-vs-rest binary classifiers, though both are valid.
- **Why not accuracy alone**: for imbalanced classes, precision/recall/F1
  and ROC-AUC tell a fuller story than accuracy — relevant given W2D3's
  fraud detection work.

## Self-review checklist

- [x] Code runs end-to-end with no errors
- [x] Comments explain _why_, not just _what_
- [x] Features scaled before fitting (gradient-based model)
- [x] Used classification metrics, not regression metrics
- [x] Reproducible (`random_state` fixed throughout)
- [x] Output evidence + plots captured
- [x] Committed with descriptive messages, pushed to feature branch

## Author

Siriyala Nishar
