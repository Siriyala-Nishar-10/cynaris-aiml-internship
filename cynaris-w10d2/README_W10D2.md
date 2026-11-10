# W10D2 — LangGraph State Machines & Conditional Edges

## Included

- Classifies five sample requests as technical, account, or general.
- Routes each category through a LangGraph conditional edge.
- Pauses response generation with `interrupt()` for human review.
- Resumes the same thread with `Command(resume=...)`.
- Uses `MemorySaver` for local in-memory checkpoints.

## Graph flow

`START -> classify -> conditional route -> category-specific respond -> END`

The three logical stages are classify, route, and respond. The route is implemented as a conditional-edge function. There are three category-specific response destination nodes so each route can have distinct behavior.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run five examples and interact with the human-review pause

```powershell
python run_demo.py
```

For each example, inspect the draft and type approval or feedback to resume.

## Run tests

```powershell
python -m pytest -q
```

## Evidence

Capture screenshots of five classifications, a paused human-review state, a resumed response, and passing pytest output.

## Scope note

Classification uses deterministic keyword rules for easy testing; it is a learning example, not a production classifier. `MemorySaver` is in-memory and is not durable production storage.
