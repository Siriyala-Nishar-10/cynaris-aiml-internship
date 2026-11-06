# W10D1: LangGraph Stateful Agent Graph (Support Ticket Triage)

A support-ticket triage agent built as a LangGraph state machine. It classifies
a customer message, routes it, and either replies automatically or **pauses for
a human decision** and resumes from exactly where it stopped.

```
START -> classify -> route --(auto)---------> respond -> END
                       |
                       +--(human)--> human_review --> respond -> END
                                     (pauses with interrupt())
```

| Node | Job |
|---|---|
| `classify` | Label the message: billing, technical, refund, general, or unknown |
| `route` | Decide `auto` or `human`, and record *why* |
| `human_review` | Pause via `interrupt()`; resume with the person's decision |
| `respond` | Write the reply (deterministic templates) |

The **conditional edge** after `route` picks `respond` or `human_review`. A message
goes to a human when it is a refund (money), when nothing matched (unknown), or
when the classifier is unsure (confidence below 0.6, e.g. a message mixing two topics).

## Run it

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt

python -m triage_agent demo          # the 5 lesson inputs, with the path each takes
python -m triage_agent chat          # interactive: pause, answer as the human, resume
python -m triage_agent diagram       # writes graph.mmd (Mermaid; renders on GitHub)
python -m pytest -q                  # 58 tests
```

Optional: `--llm` classifies with a local Ollama model instead of keyword rules
(`ollama pull llama3.2:3b`; override with `TRIAGE_MODEL`). If the model's reply is
not exactly one known label, it falls back to the rules.

## The 5 test inputs

| # | Message | Label | Route | Path |
|---|---|---|---|---|
| 1 | I was charged twice on my last invoice | billing | auto | classify > route > respond |
| 2 | The app crashes every time I open it | technical | auto | classify > route > respond |
| 3 | I want a refund, this product is broken | refund | human | classify > route > **paused** |
| 4 | Hello, what are your opening hours? | general | auto | classify > route > respond |
| 5 | asdf qwerty | unknown | human | classify > route > **paused** |

Each state records a `trace` (the nodes it visited, built with a reducer), so tests
assert the exact path instead of guessing from reply text.

## Design decisions

- **Rules first, LLM optional.** The default classifier is deterministic, so the
  routing logic is testable and free. The LLM classifier plugs in behind the same
  interface and reports `confidence=None`, because an LLM gives no calibrated
  confidence and inventing one would be dishonest. The router treats `None` as
  "no signal", not "low".
- **Fail-safe approval.** Only an explicit `{"approved": True}` (or `True`) approves.
  A typo like `"yes"` counts as a denial, so it can never trigger a refund.
- **Safe default checkpointer.** See the gotcha below.
- **One `thread_id` per ticket.** State is saved per thread; reusing an id would
  carry the previous ticket's trace into the next.

## Human-in-the-loop: what to know

1. **A checkpointer is mandatory.** Without one, `interrupt()` appears to pause but
   resuming raises `RuntimeError: Cannot use Command(resume=...) without checkpointer`
   (checked on langgraph 1.2.19 and guarded by a test). `build_graph` defaults to an
   in-memory saver for that reason.
2. **The interrupted node re-runs from its first line on resume.** Anything above
   `interrupt()` executes twice, so keep it free of side effects. Earlier nodes do
   *not* re-run (tested: the classifier is called once).
3. **`MemorySaver` lives in the process.** Restart the program and every paused
   ticket is lost. A real deployment needs a persistent checkpointer.

## Known limitations

- **Keyword rules ignore negation.** "I do NOT want a refund" is classified as a
  refund with confidence 1.0. For refunds the human gate contains the damage; for
  billing/technical it would produce a wrong automatic reply. This is the main reason
  to use the `--llm` classifier, or a better one, in practice.
- Replies are templates, not generated text. The lesson is about graph mechanics.
- Keywords are English only.

## What was and was not verified

Verified (Python 3.12.3, langgraph 1.2.19):
- 58 tests pass, and deliberately breaking the code in 6 ways (threshold off-by-one,
  loose approval, refunds skipping review, no checkpointer, reversed priority, loose
  LLM parsing) makes the suite fail each time.
- The interactive console path (pause, human input, resume) run end to end, with
  keyboard input scripted.

**Not** verified:
- Python 3.14. langgraph declares `>=3.10` but I could not test 3.14. If installing
  fails there, use a Python 3.11/3.12 virtual environment.
- The `--llm` path against a real Ollama server (unit-tested with a fake HTTP client only).
- A persistent checkpointer.

## Layout

```
triage_agent/  state.py  classify.py  graph.py  responses.py  llm_client.py  cli.py
tests/         test_classify.py  test_graph_routing.py  test_human_in_loop.py  test_cli.py
graph.mmd      generated diagram
```
