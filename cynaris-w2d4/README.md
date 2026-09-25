# W2D4: Train/Test Split & Cross-Validation (+ Feature Scaling)

## Objective

Demonstrate why distance-based and gradient-based algorithms need scaled
features, and compare `StandardScaler`, `MinMaxScaler`, and `RobustScaler`
using proper train/test splitting and cross-validation.

## Approach

1. **Dataset:** `sklearn.datasets.load_wine` — 13 numeric features on very
   different scales (e.g. `proline` in the hundreds vs `hue` under 2),
   which makes the effect of scaling obvious.
2. **Train/test split:** `train_test_split` with `stratify=y` and a fixed
   `random_state=42`, so class proportions are preserved and results are
   reproducible.
3. **Cross-validation:** 5-fold `StratifiedKFold` via `cross_val_score`.
   Scaling is done **inside** a `Pipeline` per fold (not once up front) to
   avoid data leakage — the scaler is fit only on each fold's training
   split, never on the validation fold.
4. **Models compared:**
   - `KNeighborsClassifier` — distance-based, should be very sensitive to
     unscaled features (Euclidean distance dominated by large-magnitude
     columns).
   - `SGDClassifier` — gradient-based, should converge poorly/unstably on
     unscaled data.
5. **Scalers compared:** No scaling (baseline), `StandardScaler`,
   `MinMaxScaler`, `RobustScaler`.

## Results (5-fold CV accuracy on training set)

| Model                          | Scaler         | Mean Accuracy | Std    |
| ------------------------------ | -------------- | ------------- | ------ |
| KNN (distance-based)           | No Scaling     | 0.7047        | 0.0386 |
| KNN (distance-based)           | StandardScaler | 0.9507        | 0.0353 |
| KNN (distance-based)           | MinMaxScaler   | 0.9576        | 0.0262 |
| KNN (distance-based)           | RobustScaler   | 0.9507        | 0.0476 |
| SGDClassifier (gradient-based) | No Scaling     | 0.5724        | 0.1633 |
| SGDClassifier (gradient-based) | StandardScaler | 0.9719        | 0.0344 |
| SGDClassifier (gradient-based) | MinMaxScaler   | 0.9650        | 0.0440 |
| SGDClassifier (gradient-based) | RobustScaler   | 0.9931        | 0.0138 |

## Key takeaway

Both models improve dramatically with scaling. KNN goes from ~70% to
~95-96% because distance calculations were previously dominated by
high-magnitude features. SGDClassifier goes from ~57% (high variance) to
~97-99% because gradient descent converges far more reliably when all
features share a comparable scale.

## When to use which scaler

- **StandardScaler** — default choice; assumes no extreme outliers.
- **MinMaxScaler** — bounds features to [0,1]; good when a fixed range is
  needed (e.g. neural nets), but sensitive to outliers.
- **RobustScaler** — uses median/IQR instead of mean/std; best when the
  data has outliers, as seen here with SGDClassifier's best score.

## Self-review checklist

- [x] Code runs end-to-end with no errors
- [x] Comments explain _why_, not just _what_
- [x] No data leakage (scaling done inside CV pipeline, fit only on train)
- [x] Reproducible (`random_state` fixed throughout)
- [x] Output evidence captured
- [x] Committed with descriptive messages, pushed to feature branch

# Author

Siriyala Nishar
