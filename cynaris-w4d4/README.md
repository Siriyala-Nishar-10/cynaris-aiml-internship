# W4D4 FastAPI Model Serving Endpoint

A minimal, production-shaped example of serving a scikit-learn classifier
through FastAPI, trained on an **imbalanced** binary classification problem.

```
fastapi_model_serving/
├── train_model.py          # trains + saves the model, scaler, metadata
├── requirements.txt
├── model/                  # created by train_model.py
│   ├── model.joblib
│   ├── scaler.joblib
│   └── metadata.json
├── app/
│   ├── main.py              # FastAPI app: /health, /predict, /predict/batch
│   └── schemas.py           # pydantic request/response models
└── tests/
    ├── check_predict_logic.py  # sanity check, no fastapi needed
    └── test_api.py              # pytest + TestClient suite
```

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the model

```bash
python train_model.py
```

This generates a synthetic imbalanced dataset (~8% positive class) and
prints metrics, then writes `model/model.joblib`, `model/scaler.joblib`,
and `model/metadata.json`. **Swap `make_imbalanced_dataset()` in
`train_model.py` for your own data loading** — everything else works
unchanged as long as `X` has 10 numeric columns and `y` is binary.

## 3. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Open **http://127.0.0.1:8000/docs** for interactive Swagger UI, or use curl:

```bash
# Health check
curl http://127.0.0.1:8000/health

# Single prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, -1.2, 0.3, 2.1, -0.4, 0.0, 1.1, -0.9, 0.2, 0.7]}'

# Batch prediction
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"instances": [[0.5,-1.2,0.3,2.1,-0.4,0.0,1.1,-0.9,0.2,0.7],
                      [0,0,0,0,0,0,0,0,0,0]]}'
```

## 4. Run tests

```bash
python tests/check_predict_logic.py   # no fastapi/httpx required
pytest tests/test_api.py -v           # full API test suite
```

---

## Design decisions (for the viva)

**Why class weighting + threshold tuning instead of SMOTE?**
`class_weight="balanced"` reweights the loss function without inventing or
duplicating rows, so it's a good first move on tabular data. It changes the
model's _training_ behaviour but not its output scale, so the default 0.5
cutoff on `predict_proba` is usually wrong for imbalanced problems — sweeping
thresholds on a held-out validation split and picking the one that maximises
F1 recovers most of the benefit that resampling would otherwise chase. That
tuned threshold is saved in `metadata.json` and applied at serving time
instead of hard-coding 0.5 in the API.

To compare against oversampling, install `imbalanced-learn` and swap the
`RandomForestClassifier.fit(...)` call for an `imblearn.pipeline.Pipeline`
with `SMOTE()` before the classifier — the rest of the serving code doesn't
need to change, since it only depends on `predict_proba` existing.

**Why load the model once at startup instead of per-request?**
`app/main.py` uses FastAPI's `lifespan` context manager to load
`model.joblib` and `scaler.joblib` into memory a single time when the
process starts, not on every request. Deserializing a model from disk on
every call would add latency and disk I/O proportional to traffic; loading
once means predictions only pay for the (cheap) `transform` + `predict_proba`
call.

**Why a separate `/predict/batch` endpoint?**
Calling `/predict` once per row forces N HTTP round trips for N predictions.
`/predict/batch` accepts a list of rows and returns a list of results in one
call, which matters once a client needs to score more than a handful of
rows at a time.

**What would I improve with one more day?**

- Add request logging + a `/metrics` endpoint (e.g. Prometheus counters for
  prediction volume and latency).
- Add input schema validation against `metadata.json`'s `feature_names`
  instead of a hard-coded `N_FEATURES` constant in `schemas.py`.
- Containerize with a `Dockerfile` and add a `/predict` load test.
- Track training runs with MLflow (per the approved AI/ML stack) instead of
  a flat `metadata.json` file, so threshold/metric history is queryable.

## Author

Siriyala Nishar
