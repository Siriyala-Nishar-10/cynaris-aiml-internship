# W11D5 — Tracked & Evaluated RAG Pipeline

## Objective

Build a fresh RAG application integrating LangGraph orchestration, ChromaDB retrieval, Ollama generation, a CrewAI answer reviewer, Ragas evaluation, and MLflow tracking.

## Architecture

1. LangGraph coordinates retrieval and answer generation.
2. ChromaDB stores and retrieves knowledge-base chunks using Ollama embeddings.
3. Ollama `llama3.2` generates a context-grounded answer.
4. CrewAI runs a separate reviewer to flag unsupported responses.
5. Ragas evaluates faithfulness, answer relevancy, context precision, and context recall.
6. MLflow records model/configuration parameters, answer artifacts, review output, and Ragas metrics.

## Requirements

- Python 3.11 recommended
- Ollama installed and running
- Models available locally:
  - `ollama pull llama3.2`
  - `ollama pull nomic-embed-text`

## Windows PowerShell setup

```powershell
cd cynaris-w11d5
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run one RAG question

```powershell
python main.py --question "How do I reset my password?"
```

## Run Ragas evaluation

```powershell
python main.py --evaluate
```

Ragas evaluation may take several minutes because it calls the local model. If model calls time out, verify Ollama is running and test one question first. The evaluation output is written to `results/ragas_scores.csv`.

## View MLflow

```powershell
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open `http://127.0.0.1:5000`.

## Tests

```powershell
python -m pytest -q
```

The included tests validate the project data files. They do not require Ollama.

## Suggested screenshots

1. `01_project_structure.png` — project tree in VS Code.
2. `02_rag_output.png` — question, answer, retrieved context and CrewAI review.
3. `03_mlflow_run.png` — MLflow run parameters and artifacts.
4. `04_ragas_results.png` — evaluation CSV/table with metric columns.
5. `05_pytest.png` — passing test output.
