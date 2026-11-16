# W8D4 — Documentation, Testing & Code Review

## What This Does

Reviews the W8D3 Haystack search API, fixes what the review finds, expands the
test suite, and documents the result. The service is the same FastAPI + Haystack
search API, now v1.1.0.

## What Changed

| Area                | Before (W8D3)                                                                                 | After (W8D4)                                                        |
| ------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Tests               | 6                                                                                             | 42 test cases (API, config, search)                                 |
| Config              | `os.getenv` spread across modules, unvalidated                                                | one validated `Settings` in `app/config.py`                         |
| Bugs and robustness | whitespace query accepted, unsafe lazy start-up, working-directory-dependent paths, bare 500s | all fixed, each with a test                                         |
| Documentation       | README only                                                                                   | README, `CODE_REVIEW.md`, `docs/ARCHITECTURE.md`, `docs/TESTING.md` |

Full list of findings: [CODE_REVIEW.md](CODE_REVIEW.md).

## Run

```bash
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pytest --cov=app --cov-report=term-missing
ruff check .
uvicorn app.main:app --reload
```

## Endpoints

| Endpoint                              | Purpose                             |
| ------------------------------------- | ----------------------------------- |
| `GET /health`                         | liveness check                      |
| `GET /info`                           | documents, chunks and enabled modes |
| `GET /search?q=...&mode=bm25&top_k=3` | ranked chunks with source and score |

Status codes: 200 success, 422 invalid input, 501 dense mode not enabled, 500
unexpected error (generic message, details only in the logs).

## Documentation Index

- [CODE_REVIEW.md](CODE_REVIEW.md) — findings, fixes and the review checklist
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components, request flow, configuration, limits
- [docs/TESTING.md](docs/TESTING.md) — test strategy and how to run it

## Results

- Tests: **42 passed** (`python -m pytest -q`): 15 API, 17 config, 10 search
- Coverage: **84%** of `app/` (`--cov=app`)

| File            | Coverage | Not covered                                                                                 |
| --------------- | -------- | ------------------------------------------------------------------------------------------- |
| `app/config.py` | 100%     | –                                                                                           |
| `app/main.py`   | 92%      | lines 39-41 and 46-47: the real lazy start-up (the tests inject a fake service)             |
| `app/search.py` | 69%      | lines 46-73 and 101: the optional dense retrieval path (needs PyTorch, so it is not tested) |

- Lint: `ruff check .` -> All checks passed

Evidence: `docs/tests_coverage.png` (pytest, coverage and ruff output).

## Known Limitations

- Dense retrieval is untested: it needs PyTorch and is off by default. It also duplicates the chunks in memory.
- No authentication or rate limiting.
- The index is rebuilt on every start.

## If I Had One More Day

- Add a test that starts the app with the real service (covers the start-up lines) and run tests and lint in CI with a coverage threshold
- Add a load test to check behaviour under concurrent requests
- Add request IDs to the logs so one request can be traced
- Add an authenticated endpoint to re-index uploaded documents

## Author

Siriyala Nishar
