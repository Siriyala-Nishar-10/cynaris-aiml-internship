# W3D4: SVM & KNN — When to Use What

## Objective
Compare Support Vector Machines (linear vs RBF kernel) and K-Nearest
Neighbors on a non-linearly-separable dataset, tune both, visualize
their decision boundaries, and understand the practical trade-offs
between them.

## What's in this folder
- `w3d4_svm_knn.py` — main script
- `svm_kernel_comparison.png` — linear vs RBF kernel SVM boundaries
- `svm_vs_knn_boundaries.png` — tuned SVM vs tuned KNN boundaries
- `output_evidence.txt` — captured console output

## Approach

### 1. Dataset
Used `make_moons` (2 interleaving crescents, with noise) instead of a
linearly-separable toy dataset. This is the classic case where a linear
model fails and non-linear methods (RBF-kernel SVM, KNN) are needed —
directly relevant to "when to use what."

### 2. Scaling
Both SVM and KNN are distance-based (SVM's kernel relies on dot
products/distances; KNN directly computes distances), so features were
scaled with `StandardScaler` first — same principle as W2D4.

### 3. Kernel comparison
Compared `linear` vs `rbf` kernel SVM directly:
| Kernel | Test Accuracy |
|--------|---------------|
| Linear | 0.9200        |
| RBF    | 0.9500        |

RBF wins because the moons dataset isn't linearly separable — the RBF
kernel implicitly maps data into a higher-dimensional space where a
linear separator becomes possible.

### 4. Hyperparameter tuning
- **SVM (RBF):** `GridSearchCV` over `C` (regularization strength) and
  `gamma` (kernel width). Best: `C=100, gamma=1`, CV accuracy 0.9267.
- **KNN:** `GridSearchCV` over `n_neighbors` and `weights`. Best:
  `n_neighbors=11, weights='distance'`, CV accuracy 0.9367.

### 5. Final comparison
| Model            | Test Accuracy |
|-------------------|---------------|
| SVM (RBF, tuned)  | 0.89          |
| KNN (tuned)       | 0.92          |

### 6. Timing
| Model            | Train Time (s) | Predict Time (s) |
|-------------------|----------------|-------------------|
| SVM (RBF, tuned)  | 0.0019         | 0.0002            |
| KNN (tuned)       | 0.0005         | 0.0007            |

KNN has near-zero training cost (it just stores the data — a "lazy
learner") but pays for it at prediction time since it must scan
neighbors for every query. SVM pays training cost upfront but predicts
fast, since the decision only depends on the learned support vectors.
(At this small dataset size the difference is negligible in absolute
terms, but the *pattern* holds and grows with dataset size.)

## When to use what

**SVM (RBF kernel):**
- Works well in high-dimensional feature spaces
- Robust when there's a clear margin between classes
- Fast predictions (only depends on support vectors)
- Slower to train on large datasets; less interpretable
- Needs careful tuning of `C` and `gamma`

**KNN:**
- Simple, intuitive, no real training phase
- Naturally handles multi-class and non-linear boundaries
- Slow predictions on large datasets (scans all neighbors every time)
- Suffers in high dimensions (curse of dimensionality)
- Sensitive to unscaled/irrelevant features and choice of `k`

## Self-review checklist
- [x] Code runs end-to-end with no errors
- [x] Comments explain *why*, not just *what*
- [x] Used a non-linearly-separable dataset to make kernel choice matter
- [x] Scaled features before fitting (both models are distance-based)
- [x] Tuned both models via GridSearchCV
- [x] Compared timing, not just accuracy
- [x] Reproducible (`random_state` fixed throughout)
- [x] Output evidence + plots captured
- [x] Committed with descriptive messages, pushed to feature branch

## Author
Siriyala Nishar
