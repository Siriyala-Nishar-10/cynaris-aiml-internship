# W3D3: Decision Trees & Random Forests

## Objective

Understand how trees split data using Gini impurity and information gain
(entropy), visualize a tree, see and fix overfitting via tuning, and
compare a single tree against a Random Forest.

## What's in this folder

- `w3d3_decision_trees.py` — main script
- `decision_tree_visualization.png` — the tuned tree, fully drawn out
- `tree_feature_importance.png` — top features by importance (single tree)
- `rf_feature_importance.png` — top features by importance (Random Forest)
- `output_evidence.txt` — captured console output

## Approach

### 1. Dataset

Used a synthetic dataset (`make_classification`, 800 samples, 20
features, 3 classes, 8% label noise) instead of a toy dataset like
Iris/Wine. Those are too small and clean to show a real overfitting
gap — an unrestricted tree gets 94-100% test accuracy on them
regardless of tuning, which defeats the point of the lesson. The noisy
synthetic set gives an unrestricted tree actual room to memorize noise.

### 2. Splitting criteria: Gini vs Entropy

Trained a tree with each criterion and compared depth, leaf count, and
train accuracy. Both produce very similar trees in practice — Gini is
slightly cheaper to compute (no logarithm), entropy is marginally more
sensitive to changes in class probability.

### 3. Overfitting demo

- **Unrestricted tree:** depth 11, **100% train accuracy, 80% test
  accuracy** — a clear overfitting gap; it memorized training noise.
- **Depth-limited tree (max_depth=3):** 78.3% train, 79.4% test — much
  smaller gap, better generalization, at a small cost to raw train fit.

### 4. Hyperparameter tuning

`GridSearchCV` (5-fold) over `max_depth`, `min_samples_split`, and
`min_samples_leaf` — the three main levers for controlling tree
complexity and overfitting.

**Best params:** `max_depth=5, min_samples_leaf=1, min_samples_split=2`
**Best CV accuracy:** 0.7844

### 5. Random Forest comparison

Trained a 200-tree Random Forest (bootstrap samples + random feature
subsets per split) and compared to the tuned single tree.

| Model                     | Test Accuracy |
| ------------------------- | ------------- |
| Tuned Decision Tree       | 0.8125        |
| Random Forest (200 trees) | 0.8313        |

Random Forest edges out the single tree by averaging away the variance
any one tree would have — at the cost of losing the single readable
tree diagram (feature importances are still available, but the "why"
of one prediction is no longer a simple path down a tree).

## Key concepts

- **Gini impurity**: measures how mixed the classes are at a node
  (0 = pure). Trees pick the split that most reduces weighted Gini
  impurity across child nodes.
- **Information gain (entropy)**: alternative impurity measure based
  on Shannon entropy; picks splits that most reduce entropy.
- **Overfitting in trees**: an unrestricted tree keeps splitting until
  nodes are pure (or tiny), which means it can carve out a leaf for
  individual noisy points — great train accuracy, poor generalization.
- **Controlling overfitting**: `max_depth`, `min_samples_split`,
  `min_samples_leaf`, and `max_leaf_nodes` all cap how much a tree can
  contort itself to fit training noise.
- **Random Forest**: trains many trees on bootstrapped samples with
  random feature subsets per split, then averages/votes — this reduces
  variance (overfitting) compared to any single deep tree.

## Self-review checklist

- [x] Code runs end-to-end with no errors
- [x] Comments explain _why_, not just _what_
- [x] Used a dataset that actually demonstrates overfitting (not a toy set)
- [x] Compared Gini vs entropy explicitly
- [x] Tuned via GridSearchCV rather than manual guessing
- [x] Compared single tree vs Random Forest with real numbers
- [x] Reproducible (`random_state` fixed throughout)
- [x] Output evidence + plots captured
- [x] Committed with descriptive messages, pushed to feature branch

## Author

Siriyala Nishar
