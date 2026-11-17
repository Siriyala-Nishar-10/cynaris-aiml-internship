# W9D4: CrewAI + LangChain — Hybrid Agent Systems

A practical implementation of the W9D4 requirements using LangChain and a locally running Ollama model.

## Included tasks

1. **Chain:** `PromptTemplate -> ChatOllama -> StrOutputParser`, tested with five inputs.
2. **Memory:** `ConversationBufferMemory` across five turns, including follow-up questions.
3. **Agent:** ReAct agent with two tools:
   - `web_search_stub`: sample results only; it does not access the live web.
   - `calculator`: safe arithmetic parser (does not use Python `eval`).
4. **Evidence:** outputs are written to the `results/` directory.

## Requirements

- Python 3.10 or 3.11 recommended
- Ollama installed and running
- Model `llama3.2` pulled locally

## Setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3.2
```

If `py -3.11` is unavailable, create the virtual environment with your installed supported Python version.

## Run

```powershell
python main.py
```

The first run may take longer while Ollama loads the model. The three sections run sequentially. Check:

- `results/chain_outputs.txt`
- `results/memory_outputs.txt`
- `results/agent_outputs.txt`

## Notes

- The web-search tool is deliberately a stub for this learning exercise. It returns clearly labeled sample information and does not claim to perform live searches.
- The agent uses the ReAct pattern: it chooses a tool, observes the result, and then produces a final answer.
- `ConversationBufferMemory` and `LLMChain` are classic LangChain APIs included because the lesson explicitly asks for them. They are deprecated in newer LangChain versions; the pinned dependencies keep the example on a compatible 0.3.x API line.
- This project demonstrates LangChain agent functionality. CrewAI and LangGraph are part of the broader approved stack but are not required by the listed practical tasks.

## Git workflow

Replace `your-name` with your branch identifier:

```bash
git checkout -b feat/aiml-W9-your-name
git add .
git commit -m "feat: implement langchain chain and memory"
git add .
git commit -m "feat: implement two-tool agent and tests"
git push -u origin feat/aiml-W9-your-name
```

Open a Pull Request from that branch.

## Evidence and self-review

Run the project and capture terminal screenshots showing:
- five chain inputs and outputs,
- five memory turns and the stored history,
- three agent tasks and their final answers.

Save screenshots in `screenshots/`. Complete `SELF_REVIEW.md` before raising the PR.
