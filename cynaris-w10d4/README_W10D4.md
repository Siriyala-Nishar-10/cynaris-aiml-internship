# W10D4: Human-in-the-Loop with LangGraph

## Objective

Build a three-node graph (`classify -> route -> respond`) with classification-based conditional routing and a human-review interrupt.

## Structure

```text
cynaris-w10d4/
├── screenshots/
├── src/graph.py
├── tests/test_graph.py
├── .gitignore
├── README_W10D4.md
├── requirements.txt
└── main.py
```

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

Each request pauses for review. Enter `Approved` or provide revision feedback to resume.

## Tests

```powershell
python -m pytest -q
```

Tests cover five category/route examples, pause/resume, and revision feedback.

## Design

- `classify` identifies technical, account, or general requests.
- `route` maps the category to its support route.
- A conditional edge checks the classification output before entering `respond`.
- `interrupt()` pauses execution; `Command(resume=...)` resumes the same thread.
- `MemorySaver` stores checkpoints in process memory only; they do not survive application restarts.

## Evidence

Save screenshots of the five route results, a human-review pause/resume, and passing pytest output in `screenshots/`.
