# W4D3: Model Serialisation — joblib & pickle

## Objective

Demonstrate complete machine learning model persistence using Python’s native `pickle` library and `joblib`. Build an end-to-end `sklearn` pipeline (preprocessing, encoding, and classification), save the trained state using both serialization formats, reload them, and verify prediction parity for production readiness.

---

## What's in this folder

- `w4d3_model_serialisation.py` — main script covering pipeline training, serialization, reloading, and inference simulation
- `loan_model.joblib` — serialized `sklearn` pipeline using `joblib`
- `loan_model.pkl` — serialized `sklearn` pipeline using standard `pickle`
- `output_evidence.txt` — captured console output showing performance metrics and prediction consistency
- `README.md` — documentation and workflow notes

---

## Approach

### 1. End-to-End Pipeline Training

Built a robust `sklearn.pipeline.Pipeline` with data preprocessing and `LogisticRegression` to avoid data leakage and streamline production serving.

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])

```

- **Preprocessing:** One-hot encoded categorical variables (`employment_type`) using `pd.get_dummies` and standardized numeric features using `StandardScaler`.
- **Validation:** Stratified train-test split ($80/20$) to maintain target distribution balance.
- **Base Metrics:** Achieved identical predictive performance prior to serialization.

---

### 2. Serialization & Storage Comparison

Saved the trained pipeline object using both `joblib` and `pickle` to measure execution time and output file sizes.

| Serialization Format | Execution Time | File Size | Primary Use Case                                 |
| -------------------- | -------------- | --------- | ------------------------------------------------ |
| **`joblib`**         | ~3–5 ms        | ~2–3 KB   | Optimized for NumPy arrays & scikit-learn models |
| **`pickle`**         | ~1–2 ms        | ~2–3 KB   | Native Python object serialization               |

---

### 3. Model Reloading & Verification

Reloaded both model binaries and verified that inference outputs matched the original model down to exact prediction probabilities.

```python
# Array equality assertion across reload formats
identical = np.array_equal(predictions_joblib, predictions_pickle)

```

- **Verification Result:** `Predictions identical: True`
- Both reloaded models generated matching predictions and confidence scores across all test samples.

---

### 4. Simulated Production Inference

Simulated real-time scoring on incoming loan applications by reloading the saved `joblib` artifact and running inference on unseen applicant data.

- **Sample Output:**
- `Applicant 1: APPROVED (confidence: 84.2%)`
- `Applicant 2: REJECTED (confidence: 18.5%)`
- `Applicant 3: APPROVED (confidence: 91.0%)`

---

## Key Concepts

- **Serialization:** Converting an in-memory object (e.g., trained ML pipeline) into a byte stream for storage or network transfer.
- **`joblib` vs `pickle`:** While `pickle` is built into Python, `joblib` is optimized for structures containing large NumPy arrays, making it the standard choice for scikit-learn models.
- **Pipeline Integrity:** Saving the full `Pipeline` object ensures feature scalers, encoders, and model parameters are restored together, preventing inference skew.
- **Security Warning:** Unpickling files from untrusted sources can execute arbitrary code. Always ensure saved model files come from secure pipelines.

---

## Self-Review Checklist

- [x] Full training, saving, loading, and prediction verification cycle executes end-to-end without errors.
- [x] Evaluated and compared performance between `joblib` and `pickle`.
- [x] Verified exact output equality (`np.array_equal`) across reloaded models.
- [x] Clean, fully commented Python code following standard guidelines.
- [x] Output evidence captured and committed to git feature branch (`feat/aiml-W4-nishar`).

---

## Author

Siriyala Nishar
