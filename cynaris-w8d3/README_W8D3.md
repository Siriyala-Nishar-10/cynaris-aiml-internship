# W8D3 — Deployment: Haystack Search API on Railway

## What This Does

A FastAPI service around a Haystack retrieval pipeline, packaged in Docker and
deployed on Railway. It indexes plain-text documents at start-up and returns the
best-matching chunks for a query with BM25. Dense retrieval is available as an
optional mode for local use.

Pipeline: .txt files -> DocumentCleaner -> DocumentSplitter (150 words, 20 overlap) -> InMemoryDocumentStore -> BM25 retriever -> JSON response

## Endpoints

| Endpoint | Purpose |
| -------- | ------- |
| `GET /health` | liveness check used by Railway |
| `GET /info` | number of documents, chunks and enabled modes |
| `GET /search?q=...&mode=bm25&top_k=3` | ranked chunks with source file, score and latency |

Invalid input is rejected with a 422 (query shorter than 2 characters, unknown
mode, `top_k` outside 1–10). `mode=dense` returns a 501 unless dense retrieval is enabled.

## Run Locally

```bash
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
uvicorn app.main:app --reload
```

Try `http://localhost:8000/search?q=term frequency saturation` and
`http://localhost:8000/docs`.

## Run with Docker

```bash
docker build -t haystack-api .
docker run --rm -p 8000:8000 haystack-api
```

## Deploy on Railway

1. Push the branch to GitHub.
2. In Railway: New Project -> Deploy from GitHub repo -> select the repository.
3. In the service Settings, set **Root Directory** to `cynaris-w8d3`, because the
   repository holds many projects in sub-folders.
4. Railway builds the `Dockerfile` (see `railway.json`) and injects the `PORT`
   variable; the container listens on `${PORT:-8000}`. The health check path is `/health`.
5. Settings -> Networking -> Generate Domain, then open `/health` and `/search`.

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `DOCS_DIR` | `sample_docs` | folder of `.txt` files to index |
| `ENABLE_DENSE` | `0` | set to `1` (and install `requirements-dense.txt`) to enable `mode=dense` |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | embedding model for dense mode |

## Design Decisions

| Decision | Choice | Why |
| -------- | ------ | --- |
| Reader | none, the API returns retrieved chunks | the extractive reader needs PyTorch and transformers, which are too heavy for a small Railway instance |
| Retriever | BM25 by default | light, fast, needs no model download, strong when the query shares words with the answer |
| Dense mode | optional (`ENABLE_DENSE=1`) | pulls in PyTorch; kept for local use and returns 501 when off |
| Index | in memory, built at start-up | tiny corpus; the health check passes only after the index is ready |
| Sample data | 5 short original documents in `sample_docs/` | the deployed demo works without copyrighted papers |
| Testability | the service is injected into `create_app` | API tests use a fake service and need no Haystack index |
| Container | slim Python image, non-root user, dependency layer cached | smaller image, basic hardening |

## Results

- Tests: 6 passed (5 API tests with a fake service, 1 real BM25 test that checks
  the right sample document comes first).
- Lint: `ruff check .` -> [passed after `ruff check . --fix`]
- Live deployment: [https://your-service.up.railway.app]
- `/search?q=Railway PORT environment variable` returned `deploying_apis.txt` first: [confirm]

### BM25 vs dense retrieval (from my W7D2 experiment)

Precision@3 on the same 10 questions over 5 research papers, judged by hand:

| Retriever | Precision@3 |
| --------- | ----------- |
| BM25 | 0.53 |
| Hybrid (BM25 + dense, RRF) | 0.50 |
| Dense | 0.40 |

BM25 and hybrid were effectively tied (one chunk apart over 10 questions) and
dense was lower on these fact-lookup questions, so BM25 is a sound default for
a lightweight deployment. The numbers come from that experiment; this service
exposes the same BM25 retrieval over a smaller sample corpus.

## Evidence

- `docs/local_search.png` — `/search` result running locally
- `docs/railway_deployed.png` — successful deployment in Railway
- `docs/railway_search.png` — the live URL returning a search result

## Notes

- Railway plan limits and pricing change; check the current plan before deploying.
- Environment: Python 3.13, haystack-ai 3.x, FastAPI, Windows, per-day `.venv`.

## If I Had One More Day

- Add a hybrid mode (BM25 + dense with reciprocal rank fusion) behind a flag
- Persist the index instead of rebuilding it on every start
- Add an API key and rate limiting
- Add an upload endpoint and re-indexing without a redeploy

## Author

Siriyala Nishar
