# W9D1 — CrewAI Fundamentals: A Multi-Agent Research Crew

## What This Does

A CrewAI crew of three agents that researches a topic and writes a short briefing,
running on a local Ollama model. It is run twice, without and with live web data, to
document what web search improves.

| Agent      | Role                    | Job                                                                               |
| ---------- | ----------------------- | --------------------------------------------------------------------------------- |
| Researcher | Senior Research Analyst | finds 5-7 facts and names the source of each (has the web search tool)            |
| Writer     | Technical Writer        | turns the notes into a 400-600 word briefing for a non-technical reader           |
| Reviewer   | Fact-Checking Editor    | checks the draft against the notes, removes unsupported claims, lists the changes |

Flow (sequential process): Researcher -> Writer -> Reviewer. Each task passes its
output to the next through `context`.

## Run

```bash
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
ollama pull llama3.2

python -m research_crew --label baseline                  # no web access
python -m research_crew --label web --web tool            # researcher calls web_search itself
python -m research_crew --label prefetch --web prefetch   # search done first, results given as evidence
python -m research_crew compare baseline web              # words, links, domains, figures
```

`--topic "..."` changes the topic. `$env:LLM_MODEL="qwen2.5:3b"` changes the model.

## Design Decisions

| Decision      | Choice                                                     | Why                                                                                                      |
| ------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Process       | sequential                                                 | each step needs the previous output; simple and predictable for small local models                       |
| Model         | `ollama/llama3.2` through CrewAI's `LLM` class             | local, no API key                                                                                        |
| Web search    | DuckDuckGo via `ddgs`, wrapped as a `BaseTool`             | free, no key; failures come back as text so the crew does not crash                                      |
| Tool access   | Researcher only                                            | one agent owns research; the others work from its notes                                                  |
| Delegation    | off for all agents                                         | prevents agents handing work around in circles                                                           |
| `max_iter`    | 6                                                          | caps how long a small model can loop on one task                                                         |
| Two web modes | `tool` and `prefetch`                                      | small models are unreliable at calling tools; `prefetch` guarantees real web data reaches the researcher |
| Comparison    | simple counts (links, domains, figures) plus manual review | transparent, but only a proxy for quality                                                                |

## Results

Topic: _Running AI assistants locally with Ollama: benefits and limits for small businesses_.
Model: `llama3.2` (3B, local). Process: sequential.

| Run                     | Words | Links | Domains | Figures | Time    |
| ----------------------- | ----- | ----- | ------- | ------- | ------- |
| baseline (no web)       | 439   | 3     | 3       | 15      | 416.7 s |
| web ([tool / prefetch]) | 777   | 0     | 0       | 31      | 657.5 s |

### Baseline review

The baseline looks well sourced (links, percentages, dates) but is largely unreliable,
because the model has no way to look anything up:

- It states that Ollama is "developed by Google". That is wrong: Ollama is an independent open-source project.
- Its sources look invented (a `github.com/google/ollama` page, a "Tobe Hooper Institute" study, an April 2022 Google blog post about Ollama). Link check: [open each URL and record: opens and is about Ollama / 404 / unrelated page].
- It reports "10,000 inference requests per second" for a local tool on small-business hardware, with no evidence.
- The Researcher admitted its knowledge stops at December 2023, yet still gave dated sources.
- The Reviewer removed one unsupported claim (a "10 watts" figure) and kept everything else,
  because it can only compare the draft with the research notes, not with the real world.

So the link and figure counts measure how sourced a text _looks_, not whether it is true.
That is why the web run is checked link by link, not only counted.

### Improvement documented

[2-3 sentences, from your real results: what changed once the Researcher had live web
data, with specific examples from `outputs/baseline.md` and `outputs/web.md` (for
example which false claim disappeared and which real source replaced it), and what is
still weak (for example search snippets are short, so some facts are shallow).]

Evidence: `docs/crew_run.png` (verbose crew run), `docs/comparison.png` (compare table).

## Concepts

- **Sequential vs hierarchical:** sequential runs tasks in a fixed order; hierarchical adds a
  manager agent that plans, delegates and validates the other agents' work.
- **Agent communication:** in a sequential crew, agents share information through task
  `context` (one task's output becomes input to the next); with delegation on, agents can
  also ask each other questions.
- **CrewAI vs a single LangChain agent:** use CrewAI when the work splits into distinct
  roles that check each other's output (research, write, review); a single agent is simpler
  and cheaper when one step with a few tools is enough.

## Limitations

- The model is small (3B), so output quality varies between runs; the same topic can give
  different results each time.
- Web search can be rate limited or return irrelevant pages.
- Link and figure counts show how well-sourced a text is, not whether it is correct.

## If I Had One More Day

- Try the hierarchical process and compare cost and quality
- Add a tool that reads a full web page, not only search snippets
- Score both outputs with Ragas faithfulness against the gathered sources
- Run each mode several times to measure how much results vary

## Author

Siriyala Nishar
