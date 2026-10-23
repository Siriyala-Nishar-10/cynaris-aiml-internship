# Code Review — Haystack Search API (W8D3 -> W8D4)

**Reviewed:** `app/main.py` and `app/search.py` from the W8D3 Railway service.
**Reviewer:** self-review, using the checklist below.
**Outcome:** 8 changes made (1 behaviour bug, 3 robustness fixes, 2 maintainability
fixes, 2 test and documentation gaps). 2 limitations documented, not fixed.

## Findings

| # | Severity | Where | Finding | Fix | Status |
| - | -------- | ----- | ------- | --- | ------ |
| 1 | Medium (bug) | `main.py` `/search` | A query of only spaces (`"   "`) passes `min_length=2`, so it reached the retriever as an empty query | Strip the query first and return 422 if fewer than 2 characters remain; the stripped text is also what gets logged and returned | Fixed, tests `test_invalid_requests_are_rejected_with_422`, `test_query_is_stripped_before_searching` |
| 2 | Medium | `main.py` `get_service` | Lazy index creation had no lock, so two simultaneous first requests could build two indexes | Double-checked locking with `threading.Lock` | Fixed |
| 3 | Medium | `search.py` | A relative `DOCS_DIR` depended on the current working directory, so the app could index a different folder depending on how it was started | Resolve relative paths against the project root | Fixed, tests in `test_config.py` |
| 4 | Medium | `main.py` `/search` | An unexpected error produced a bare 500 with no log context | `logger.exception` with mode, `top_k` and query length (not the query text), and a clean `{"detail": "search failed"}` response that leaks no internals | Fixed, `test_unexpected_error_returns_clean_500` |
| 5 | Low | `search.py` | `os.getenv` calls were spread through the module, so configuration was read in several places and never validated | New `app/config.py`: one validated `Settings` object | Fixed |
| 6 | Low | config | `CHUNK_OVERLAP` greater than or equal to `CHUNK_WORDS` or a non-number would fail deep inside Haystack | Validate at start-up with a clear `ValueError` | Fixed, `test_invalid_chunk_settings_are_rejected` |
| 7 | Low | `search.py` `load_documents` | An empty folder raised a generic `RuntimeError` | Dedicated `NoDocumentsError` that names the folder | Fixed |
| 8 | Low | tests | Only the happy path and one real-index test (6 tests); no tests for validation, config or error handling | 42 test cases across API, config and search; coverage report (see `docs/TESTING.md`) | Fixed |
| 9 | Info | `search.py` dense mode | Dense mode keeps a second copy of the chunks in memory, and needs PyTorch | Documented in README as a known limit | Not changed |
| 10 | Info | whole API | No authentication or rate limiting | Documented in README "Known limitations"; out of scope for today | Not changed |

## What was already good (kept)

- The service is injected into `create_app`, so the API is testable without Haystack.
- Logs record counts and latency, not user query text.
- Input is validated by FastAPI (length, mode and `top_k` bounds).
- The Docker image runs as a non-root user and caches the dependency layer.

## Review checklist

Tick each box only after you have checked it yourself.

**Correctness**
- [ ] Every endpoint returns the documented status codes (200, 422, 501, 500)
- [ ] Edge inputs are handled: empty, whitespace, very long, unknown mode, `top_k` bounds
- [ ] No result depends on the working directory

**Error handling and security**
- [ ] Errors are logged with context and do not leak internal messages to the client
- [ ] User query text is not written to the logs
- [ ] No secrets or credentials are in the code or the repository
- [ ] Copyrighted documents are not committed (only the original sample documents)

**Code quality**
- [ ] `ruff check .` passes
- [ ] Functions have docstrings that say what, not how
- [ ] Configuration is in one place and validated

**Tests**
- [ ] `python -m pytest -q` passes
- [ ] A test exists for each finding above that was a bug or robustness fix
- [ ] Coverage was measured and the uncovered lines are understood

**Documentation**
- [ ] README explains how to run, test and deploy
- [ ] `docs/ARCHITECTURE.md` matches the code
- [ ] Known limitations are written down
