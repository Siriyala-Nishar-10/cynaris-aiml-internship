# W9D2: CrewAI Tools — Multi-Agent Research Crew with Web Search

## Objective

Define three CrewAI agents (Researcher, Writer, Reviewer), give each a task, run the crew on a research topic, then add a web search tool, run again with real web data, and document the difference.

## Scope Note

The overall topic covers "Web Search, Code Execution", but the core implementation focuses on three agents, their respective tasks, and a web search tool. Code execution is not included in this setup.

## Project Structure & Contents

- `research_crew/config.py` — Configuration for model, base URL, temperature, and iteration limits.
- `research_crew/tools.py` — Web search implementation (`web_search`, `format_results`) and CrewAI tool wrapper.
- `research_crew/crew.py` — Agent definitions, task creation, sequential crew assembly, and execution (`run_crew()`).
- `research_crew/compare.py` — Metrics evaluation logic and report generation functions.
- `research_crew/compare_runs.py` — Execution wrapper for **Task 3**: runs both pipeline versions and outputs `comparison_report.md`.
- `run_crew.py` — Command-line entry point to run a single crew execution (with or without search).
- `tests/` — Suite of 36 unit tests covering tools, formatting, metrics, and agent setups.
- `requirements.txt` — Project dependencies (`crewai`, `duckduckgo-search`, etc.).
- **Generated Outputs:** `output_no_search.md`, `output_with_search.md`, `comparison_metrics.json`, `comparison_report.md`.

## Setup

1. **Ollama Setup:** Ensure Ollama is running and the target model is pulled:
   ```powershell
   ollama pull llama3.2:3b
   ```
2. **Environment & Dependencies:**
   Activate your virtual environment and install dependencies:
   ```powershell
   .venv\Scripts\Activate
   pip install -r requirements.txt
   ```
   _(Note: If Python version conflicts arise with `crewai`, ensure your venv uses Python 3.12 or 3.13)._

## Execution Instructions

Run tests or execute the pipeline using the following commands:

```powershell
# Run 36 unit tests (offline, no LLM or network required)
python -m unittest discover -s tests -t .

# Run single baseline execution (without web search)
python run_crew.py

# Run single execution with web search enabled
python run_crew.py --search

# Run full comparison workflow (generates both outputs, metrics, and report)
python -m research_crew.compare_runs
```

_Note: Running a 3-agent crew on a local 3B model typically takes a few minutes per run. Avoid interrupting the process while tasks finish execution._

## Architecture & How It Works

- **Agents:**
  - **Research Analyst:** Gathers domain information and facts.
  - **Technical Writer:** Converts research notes into a cohesive technical report.
  - **Fact-Checking Editor:** Verifies facts, strips unsupported claims, and ensures formatting constraints.
- **Task Flow & Communication:**
  - Sequential pipeline: `Research -> Write -> Review`.
  - Context pass-through: The Writer receives context from `[research]`; the Reviewer receives context from `[research, write]`.
  - `allow_delegation` is disabled for all agents to keep execution deterministic on lightweight local models.
- **Search Tool Integration:**
  - The web search tool is assigned **only** to the Research Analyst during search-enabled runs.
  - The tool formats search results with explicit source indices and URLs.
- **Reviewer Rules:**
  - Identifies unsupported assertions and formats the output into a clean briefing.
  - Constrained against fabricating or introducing external URLs not present in the research context.

## Design Decisions

- **Anti-Fabrication Constraint:** For non-search runs, agents are explicitly instructed to note the lack of external sources rather than hallucinating URLs or citations.
- **Strict Citation Policy:** For search runs, citations must originate directly from search tool outputs.
- **Resilient Search Tooling:** Network timeouts and rate limits return error messages to the agent rather than crashing the pipeline.
- **Iteration Limits:** `max_iter=6` prevents local models from entering endless tool-use loops.
- **Decoupled Architecture:** Search formatting and metrics logic are pure functions independent of CrewAI, allowing offline unit testing.
- **Fault-Tolerant Comparison:** `compare_runs.py` handles individual run failures gracefully without invalidating the entire report.

## Testing & Validation

The test suite (36 tests) covers:

- Search result normalization, deduplication, truncation, and error handling.
- Calculation of comparative metrics and markdown generation.
- Mock/stub evaluation of CrewAI wiring (confirming agent goals, backstories, sequential flow, and search-tool assignment).

_Limit Note: Tests verify configuration structure and call patterns, but model behavior and actual prompt adherence depend on the runtime LLM._

## Results Comparison

| Metric                  | Without Search                          | With Search                |
| ----------------------- | --------------------------------------- | -------------------------- |
| **Word Count**          | ~290 words                              | ~340 words                 |
| **URLs Cited**          | 0                                       | 3                          |
| **Unique Domains**      | 0                                       | 2                          |
| **Has Sources Section** | Yes (explicitly states no sources used) | Yes (contains active URLs) |

### Manual Checks Summary

- **URL Verification:** Opened cited URLs from `output_with_search.md` to confirm they directly support technical statements regarding local RAG pipelines.
- **Baseline Hallucinations:** Confirmed `output_no_search.md` contained 0 fabricated URLs due to anti-hallucination instructions.
- **Search Impact:** Web search added current framework specifics and benchmark statistics not present in the local model's parameters.
- **Reviewer Intervention:** The reviewer removed generalized claims and aligned all citation indices strictly with research notes.

### Qualitative Assessment

Adding web search provided real-world grounding for the local LLM. It replaced generic summaries with specific, verifiable technical details and provided actual web references without hallucinating links.

## Self-Review Checklist

- [x] Tests pass (`python -m unittest discover -s tests -t .`)
- [x] Baseline run completed and saved (`output_no_search.md`)
- [x] Search run completed and saved (`output_with_search.md`)
- [x] Comparison script executed (`comparison_metrics.json` and `comparison_report.md` generated)
- [x] Cited URLs manually validated
- [x] Code structured cleanly with test coverage
