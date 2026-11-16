# Testing Strategy

## Layers

| File | What it tests | Needs Haystack? |
| ---- | ------------- | --------------- |
| `tests/test_api.py` | HTTP behaviour with a fake service: status codes, validation, stripping, empty results, error handling | no |
| `tests/test_config.py` | environment parsing and validation | no |
| `tests/test_search.py` | real Haystack indexing and BM25 ranking on the sample documents | yes |

The fake service keeps API tests fast and independent of Haystack. The real-index
tests prove the search itself works, using queries that should clearly favour
one sample document.

## Run

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q                                          # all tests
python -m pytest --cov=app --cov-report=term-missing         # with coverage
ruff check .
```

## Test cases by purpose

- **Happy path:** health, info, search returns hits with latency.
- **Input validation:** too short, only spaces, too long, unknown mode, `top_k` out of range, missing query.
- **Regression tests for review findings:** whitespace query (finding 1), config resolution and validation (3, 6), clean 500 (4), empty folder (7).
- **Behaviour:** query is stripped, `top_k` is forwarded and respected, scores are in descending order, dense mode returns 501 when disabled.

## Not covered

- Dense retrieval (needs PyTorch; tested manually when enabled).
- The Railway deployment itself (checked by hand after each deploy).
- Concurrency under load (the lock is reviewed, not stress-tested).
