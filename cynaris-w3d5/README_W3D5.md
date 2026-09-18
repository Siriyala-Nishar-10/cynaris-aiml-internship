# W3D5: Hyperparameter Tuning — GridSearch & RandomSearch

## Objective

Compare exhaustive `GridSearchCV` against `RandomizedSearchCV` on the
same model and parameter space: how many fits each requires, how long
each takes, and how close RandomizedSearchCV's result gets to
GridSearchCV's for a fraction of the compute.

## What's in this folder

- `w3d5_hyperparameter_tuning.py` — main script
- `search_progress_comparison.png` — best-score-so-far vs combinations tried
- `output_evidence.txt` — captured console output

## Approach

### 1. Model & dataset

Used `RandomForestClassifier` on a synthetic dataset (`make_classification`,
800 samples, 20 features, 5% label noise) — enough real signal that
tuning matters, but noisy enough that the best hyperparameters aren't
obvious without a search.

### 2. GridSearchCV — exhaustive search

Grid: `n_estimators` (3) × `max_depth` (3) × `min_samples_split` (3) ×
`min_samples_leaf` (2) = **54 combinations × 5-fold CV = 270 fits.**

- **Time:** 40.59s
- **Best params:** `max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=150`
- **Best CV accuracy:** 0.8734

### 3. RandomizedSearchCV — sampled search

Same 4 hyperparameters, but as continuous distributions (`scipy.stats.randint`)
rather than fixed lists, sampled **20 times × 5-fold CV = 100 fits** —
about 37% of GridSearchCV's fit count.

- **Time:** 15.75s
- **Best params:** `max_depth=17, min_samples_leaf=3, min_samples_split=13, n_estimators=84`
- **Best CV accuracy:** 0.8641

### 4. Efficiency comparison

| Method             | Total Fits | Time (s) | Best CV Accuracy |
| ------------------ | ---------- | -------- | ---------------- |
| GridSearchCV       | 270        | 40.59    | 0.8734           |
| RandomizedSearchCV | 100        | 15.75    | 0.8641           |

**RandomizedSearchCV used 37.0% of the fits GridSearchCV used, took
61% less time, and still reached 98.9% of GridSearchCV's best CV
accuracy.**

### 5. Held-out test set confirmation

| Model                         | Test Accuracy |
| ----------------------------- | ------------- |
| GridSearchCV best model       | 0.8625        |
| RandomizedSearchCV best model | 0.8562        |

Both generalize similarly — the gap between the two search methods is
much smaller in practice than the fit-count gap would suggest.

## When to use which

**GridSearchCV:**

- Guaranteed to find the best combination _within the grid you define_
- Cost grows multiplicatively with every added parameter or value
  (combinatorial explosion) — doesn't scale to wide searches
- Wastes compute exhaustively checking unpromising regions

**RandomizedSearchCV:**

- Fixed cost (`n_iter`), independent of how many parameters you search
- Can sample from continuous distributions (e.g. `loguniform` for
  something like SVM's `C`), not just fixed discrete lists
- Often reaches a near-optimal result far faster than exhaustive search
- No guarantee of finding the true best combination

**Rule of thumb:** use RandomizedSearchCV for wide, exploratory searches
across many parameters, then optionally follow up with a narrow
GridSearchCV around the best region it finds.

## Self-review checklist

- [x] Code runs end-to-end with no errors
- [x] Comments explain _why_, not just _what_
- [x] Compared fit count, time, and accuracy — not accuracy alone
- [x] Confirmed results hold on a held-out test set, not just CV
- [x] Reproducible (`random_state` fixed throughout)
- [x] Output evidence + plot captured
- [x] Committed with descriptive messages, pushed to feature branch

## Author

Siriyala Nishar
