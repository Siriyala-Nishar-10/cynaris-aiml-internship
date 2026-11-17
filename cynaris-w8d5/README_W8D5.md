# Local AI Research Assistant

W8D5 capstone on the approved 3M stack: **CrewAI · LangGraph · MLflow · Ragas · MLOps**.
Fully local: the LLM and embeddings both run through Ollama, no API keys.

| Concern | Tool | File |
|---|---|---|
| LLM + embeddings | Ollama (`llama3.2`, `nomic-embed-text`) | `crew.py`, `evaluate.py`, `memory.py` |
| Orchestration | LangGraph: retrieve -> route -> crew / fallback | `graph.py` |
| Agents | CrewAI: analyst -> writer (sequential) | `crew.py` |
| Evaluation | Ragas: faithfulness, answer relevancy | `evaluate.py` |
| Tracking | MLflow: params, latency, scores, answer artifact | `service.py` |
| Interface | stdlib HTTP server + single-page HTML | `web.py`, `static/index.html` |
| MLOps | env-var config, unit tests, GitHub Actions CI | `config.py`, `tests/`, `.github/` |

## Setup
    ollama pull llama3.2 && ollama pull nomic-embed-text
    pip install -r requirements.txt

## Run
    export PYTHONPATH=src            # Windows PowerShell: $env:PYTHONPATH="src"
    python -m research_assistant.cli ingest ./notes
    python -m research_assistant.cli ask "What are the key findings?"
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # compare runs

## Web UI
    export PYTHONPATH=src            # Windows PowerShell: $env:PYTHONPATH="src"
    python -m research_assistant.web      # open http://127.0.0.1:8000

Upload `.txt`/`.md` files (or index a folder), ask questions, and read cited
answers with expandable sources, latency and optional Ragas scores. It binds to
localhost only. Set `PORT` to change the port.

## Test (offline, no Ollama needed)
    python -m unittest discover -s tests -v

## Design decisions
- Retrieval is grounded: the crew only sees retrieved chunks and must cite `[n]`.
- Empty retrieval short-circuits to a fallback instead of letting the LLM guess.
- Everything is configurable by env var (`CHAT_MODEL`, `EMBED_MODEL`, `TOP_K`, ...).
- Library telemetry (CrewAI, Ragas) is switched off, so nothing leaves the machine.
- Every question is an MLflow run so prompt/model/chunking changes are comparable.
