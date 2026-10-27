# W8D1 — Dockerised RAG API with CI/CD and a Monitoring Strategy

## What This Does

Wraps the Week 7 RAG system in a FastAPI service, packages it in a Docker
image, and adds a GitHub Actions pipeline: lint -> test -> build image -> push
to GitHub Container Registry (GHCR). `MONITORING.md` documents what to track,
which alerts to set, and what triggers a re-index.

## Layout

```
cynaris-w8d1/
├── app/
│   ├── main.py      # FastAPI: /health, /stats, /query
│   └── rag.py       # LlamaIndex + Ollama backend and a stub backend
├── tests/test_api.py
├── Dockerfile, .dockerignore
├── requirements.txt (runtime), requirements-rag.txt (LlamaIndex), requirements-dev.txt
├── MONITORING.md
└── .github/workflows/ci.yml   # copy to the REPOSITORY ROOT
```

## Run locally (no Docker)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
$env:RAG_BACKEND="stub"; uvicorn app.main:app --reload
```

## Run with Docker

```bash
docker build -t rag-api .
# demo without Ollama (canned answers):
docker run --rm -p 8000:8000 -e RAG_BACKEND=stub rag-api
# real RAG: Ollama on the host, .txt files in ./data
docker run --rm -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 -v "${PWD}/data:/app/data" rag-api
```

Try it (PowerShell):

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post http://localhost:8000/query -ContentType "application/json" -Body '{"question":"What is RAG?"}'
```

## CI/CD Pipeline

| Job              | What it does                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| `lint`           | `ruff check .`                                                                                                     |
| `test`           | `pytest` with the stub backend (no Ollama needed)                                                                  |
| `build-and-push` | builds the image; on push events logs in to GHCR with `GITHUB_TOKEN` and pushes tags `<branch>` and `sha-<commit>` |

Jobs run in order (`needs:`), so a lint or test failure stops the build. Pull
requests build the image but do not push it.

## Design Decisions

| Decision           | Choice                                                  | Why                                                             |
| ------------------ | ------------------------------------------------------- | --------------------------------------------------------------- |
| Backend interface  | `RagBackend` with `LlamaIndexBackend` and `StubBackend` | CI tests need no Ollama; the container can be demoed without it |
| Lazy index build   | built on the first `/query`, not at import              | indexing is slow and would break tests and startup              |
| Ollama location    | outside the container, via `OLLAMA_HOST`                | models are large; the image stays small                         |
| Documents          | mounted at `/app/data`, not baked into the image        | the source papers are copyrighted                               |
| Container user     | non-root `appuser`                                      | basic container hardening                                       |
| Dependency layers  | requirements copied before code                         | Docker reuses the cached install layer                          |
| Registry           | GHCR with `GITHUB_TOKEN`                                | no extra secrets to manage                                      |
| Request validation | `question` must be 3–500 characters                     | rejects empty or oversized input with a 422                     |

## Results

Fill these in after you run everything:

| Check                              | Result                                                            |
| ---------------------------------- | ----------------------------------------------------------------- |
| `python -m pytest -q`              | 6 passed                                                          |
| `ruff check .`                     | all checks passed                                                 |
| `docker build`                     | success, image size [x MB]                                        |
| `docker run` with the stub backend | `/health` returned `{"status":"ok"}`, `/query` returned an answer |
| GitHub Actions run                 | lint, test and build-and-push all green: [link to the run]        |
| Image in GHCR                      | [ghcr.io/... tag]                                                 |

## Evidence

- `docs/docker_run.png` — container running locally with `/health` and `/query` responses
- `docs/actions_green.png` — the passing GitHub Actions run (lint, test and build-and-push)

## Notes

- pytest shows one harmless deprecation warning (Starlette's `TestClient` prefers `httpx2`). It does not affect the tests.
- Workflows only run from the repository root `.github/workflows/`. If the
  project is not at the root, set `APP_DIR` (and the `paths:` filters) in
  `ci.yml` to the folder name.
- GHCR image names must be lower case; `docker/metadata-action` lowercases them.
- The first pushed package may be private; change its visibility in GitHub
  package settings if needed.

## If I Had One More Day

- Add Prometheus metrics (`/metrics`) and a Grafana dashboard
- Run the golden set automatically in CI against the stub and nightly against Ollama
- Add a Ragas evaluation step and fail the pipeline on a quality drop
- Add a `docker compose` file that starts the API and Ollama together

## Author

Siriyala Nishar
