# W9D3: Building a Research Crew — Multi-Agent Pipeline

## Objective

Define three CrewAI agents (Researcher, Writer, Reviewer), give each a task, run
the crew on a research topic, review the output quality, then add a web search tool,
run again with real web data, and document the improvement.

## Note on this lesson's page

The four practical tasks and commit message on this page are word for word the same
as W9D2's. This folder reuses the crew from W9D2 and makes the pipeline itself the
deliverable, adding:

1. **Every stage's output is saved** (`research.md`, `draft.md`, `final.md`) so you can see where quality is gained or lost between agents.
2. **A quality rubric** on the final briefing (length, structure, Sources section, and no URLs the research never mentioned).
3. **Citation provenance:** every URL the Researcher cites is checked against the URLs the search tool actually returned. This catches an invented link even when search was available, which a plain URL count cannot.

## What's in this folder

- `research_crew/config.py` — model, base URL, temperature, iteration cap
- `research_crew/tools.py` — web search, `SearchLog` (records queries and returned URLs), CrewAI tool wrapper
- `research_crew/crew.py` — agents, tasks, crew, `run_crew()` returning a `CrewRun` (final text, per-stage outputs, duration)
- `research_crew/analysis.py` — URL metrics, quality rubric, provenance check
- `research_crew/pipeline.py` — runs one mode, saves stage files, builds the report
- `run_pipeline.py` — runs both modes and writes `pipeline_report.md`
- `tests/` — 62 tests
- Outputs _(generated when you run it)_: `runs/no_search/`, `runs/with_search/`, `pipeline_report.md`

## Setup

1. Ollama running with the model pulled: `ollama pull llama3.2:3b`.
2. `pip install -r requirements.txt` (installs `crewai` and `ddgs`).
   - If pip reports a Python-version incompatibility for crewai, your venv's Python is probably too new for it. Create a venv with Python 3.12 or 3.13 and reinstall.

## Run (from inside this folder)

```
python -m unittest discover -s tests -t .     # 62 tests, no CrewAI/Ollama/internet needed
python run_pipeline.py                        # both modes + report
python run_pipeline.py --only with_search     # just one mode (no report)
```

Each run takes several minutes on a 3B local model. Don't interrupt it.

## How it works

- **Agents:** Research Analyst, Technical Writer, Fact-Checking Editor, each with a role, goal and backstory.
- **Tasks:** research → write → review. The writer gets the research via `context=[research]`; the reviewer gets research and draft via `context=[research, write]`.
- **Process:** sequential, delegation off for every agent.
- **Search:** only the Researcher gets the tool, and only in the with-search run. Every query and returned URL is logged to `runs/with_search/search_log.json`.
- **No-search baseline:** the agents are told not to invent URLs or citations and to say no sources were consulted. Any URL that appears anyway is listed in the report.

## Design decisions

- **Provenance over URL counting.** A URL count rewards a model for printing links, real or not. Comparing cited URLs against what the search tool returned tests whether the citations are traceable.
- **The rubric and the provenance check answer different questions.** The rubric checks the final text against the research stage; provenance checks the research stage against the search results. A run can pass the first and fail the second (a fabricated citation that the Writer and Reviewer faithfully carried forward), which the tests cover.
- **Stage files are saved even though only the final text is graded,** because "where did quality change?" is the question a multi-agent pipeline exists to answer.
- **A failed run is recorded, not raised.** One flaky run on a small model doesn't lose the other; a missing CrewAI install still stops with a clear message.
- **Search and analysis are plain functions,** so they're unit tested without CrewAI.
- **All checks are labelled crude** in the report, and a manual-checks section is built in.

## Testing (and its limits)

62 tests cover: search normalisation, de-duplication, truncation and error handling; the `SearchLog`; URL normalisation; the quality rubric (pass, too short, no Sources section, too few headings, unsupported URL, cosmetic URL differences ignored); the provenance check (traceable, fabricated, repeated citations, no results); the pipeline runner and report (files written, failures recorded, missing CrewAI still raises); and the crew wiring (3 agents with role/goal/backstory, task order and context chain, sequential process, tool only on the Researcher when requested, stage-output capture and fallbacks).

**Limit:** the crew-wiring tests run against a _stub_ `crewai` that records how it's called. They show our code builds and reads the crew as intended; they do **not** prove the real CrewAI accepts these arguments, that it returns per-task outputs in the shape assumed (`tasks_output[i].raw`), or that a 3B model follows the instructions. If the stage files come out empty or only `final.md` appears, the per-task output shape in your CrewAI version differs; the code falls back to the final text only. That only shows up on a real run.

**Stage comparison** (`research.md` → `draft.md` → `final.md`): what the Writer added or dropped, and what the Reviewer changed:

## Self-review checklist

- [x] Tests pass (`python -m unittest discover -s tests -t .`)
- [x] `run_pipeline.py` ran in both modes; `runs/` and `pipeline_report.md` generated
- [x] I read the stage files, not only the final output
- [x] I opened the cited URLs rather than trusting the rubric or provenance results
- [x] Manual-checks section completed by hand; results table filled with real numbers
- [x] Code commented; 2+ commits; PR raised
