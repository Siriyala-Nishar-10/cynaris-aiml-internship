# W10D3 — LangGraph + Memory: Persistent Conversations

## Objective
Build a three-node LangGraph workflow (`classify → route → respond`) with conditional routing, human-in-the-loop review, and checkpointed conversation memory.

## Project structure
```text
cynaris-w10d3/
├── src/
│   └── graph.py
├── tests/
│   └── test_graph.py
├── results/
├── screenshots/
├── .env.example
├── .gitignore
├── main.py
├── README_W10D3.md
└── requirements.txt
```

## Features
- **Three nodes:** `classify`, `route`, and `respond`.
- **Conditional routing:** technical and account requests pass through the `route` node; general requests go directly to `respond`.
- **Human-in-the-loop:** `interrupt()` pauses response generation for review; `Command(resume=...)` resumes the graph with human feedback.
- **Persistent conversation state:** a checkpointer saves state by `thread_id`. Reusing the same thread ID across turns restores the conversation history.
- **Tests:** classification, route selection, interrupt/resume, and same-thread history.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the demo
```powershell
python main.py
```
The demo runs five inputs with one shared thread ID. At each human-review pause, type `Approved` or enter feedback.

## Run tests
```powershell
python -m pytest -q
```

## Important memory note
This project uses LangGraph's in-memory `MemorySaver` checkpointer. It preserves state across graph invocations that reuse the same `thread_id` while the Python process/checkpointer remains alive. It does **not** provide durable storage across process restarts. A persistent database checkpointer can be added if durable, restart-safe storage is required.

## Evidence to capture
1. Terminal showing all five sample inputs and their categories/routes.
2. Terminal showing `[PAUSED FOR HUMAN REVIEW]` and the resumed response.
3. Test output from `python -m pytest -q`.

Save screenshots in `screenshots/` and test/demo output in `results/`.

## Git workflow
Use the branch format required by the course:
```powershell
git switch -c feat/aiml-W10-your-name
git add src tests main.py requirements.txt README_W10D3.md .gitignore .env.example
git commit -m "feat: add LangGraph conversation workflow"
git add .
git commit -m "test: verify routing memory and human review"
git push -u origin feat/aiml-W10-your-name
```
Then open a pull request on your repository. Replace `your-name` with your own branch suffix.

## Self-review checklist
- [ ] Three graph nodes are implemented.
- [ ] Five inputs are tested.
- [ ] Conditional routing matches each classification.
- [ ] Human-review interrupt pauses execution.
- [ ] Human feedback resumes execution.
- [ ] Same `thread_id` retains conversation history across turns.
- [ ] Tests pass.
- [ ] Demo/test evidence is saved.
- [ ] At least two CIA Mentor Mode interactions are logged.
- [ ] At least two descriptive commits are pushed and a PR is opened.

## CIA interaction log
Record the actual date, prompt, and feedback from each of your two required CIA Full Stack Mentor Mode interactions. Do not mark these complete until performed.

| Interaction | Prompt/topic | Feedback/action taken | Completed |
|---|---|---|---|
| 1 | Review graph design and conditional routing | Add actual CIA feedback here | [ ] |
| 2 | Review checkpoint memory and interrupt/resume tests | Add actual CIA feedback here | [ ] |

## Viva preparation
1. **What was built?** A three-node LangGraph workflow that classifies each message, selects a route, pauses for human review, and resumes to produce a response. A checkpointer stores state under a thread ID.
2. **Hardest part?** Managing the pause/resume lifecycle and ensuring each conversation uses the same thread ID. Explain the issue you actually encountered and how you resolved it.
3. **What would you improve?** Replace in-memory checkpointing with a persistent database checkpointer, add richer classification, and strengthen automated evaluation.
