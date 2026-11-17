# W10D5 — Stateful Customer Support Agent

## Overview

A stateful customer-support workflow built with LangGraph. It classifies customer requests, retrieves FAQ context, drafts a grounded response, and pauses for human review for billing-related or high-priority requests.

## Approved stack

- **LangGraph:** state, routing, checkpointing, interrupt/resume.
- **CrewAI:** optional triage and resolution agents (`USE_CREWAI=true`).
- **MLflow:** optional experiment/run tracking (`ENABLE_MLFLOW=true`).
- **Ragas:** evaluation dataset export is included; configure an evaluator LLM and embeddings before metric scoring.
- **MLOps:** tests, reproducible dependencies, Git workflow, and CI-ready structure.

## Windows setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run and test

```powershell
python main.py
python -m pytest -q
```

Default mode is deterministic and requires no API key or Ollama installation. Enter `approve` at a review pause to approve; other feedback escalates to a support specialist.

## Optional integrations

Copy `.env.example` to `.env`. Set `USE_CREWAI=true` only after configuring and testing a supported local Ollama provider/model. Set `ENABLE_MLFLOW=true` to log runs locally. `python -m src.evaluation` exports sample evaluation rows under `results/`; Ragas metric scoring requires a configured evaluator model and embeddings.

## Design notes and limitations

- Each conversation uses a thread ID. `MemorySaver` stores checkpoints in memory only; state is not preserved after the process exits.
- FAQ retrieval is keyword-based for this demo, not production semantic search.
- The default response generator is deterministic. CrewAI is an optional LLM-powered path.
- Do not send passwords, full payment card details, or other sensitive information in support messages.
