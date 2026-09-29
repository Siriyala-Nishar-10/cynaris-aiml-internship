# W5D2: Prompt Engineering & System Prompts with Ollama

## Objective

Explore how system prompts and prompting techniques change a model's
output, using reusable prompt templates.

## Note on this lesson's page

The title, video (LangChain prompt templates with Ollama/Llama3) and
today's date make the intended focus clear: **prompt engineering and
system prompts**. The pasted Practical Tasks / commit message /
resources were identical, word-for-word, to yesterday's W5D1 page
(same 4 tasks, same `feat: ollama setup` commit message) — a
copy-paste duplication, not this day's real content.

## Tooling note

`langchain` / `langchain-ollama` could not be installed in this
sandbox (no internet access — verified with `pip install langchain
langchain-ollama`, same failure as aif360/shap in W4D6). The core
LangChain `PromptTemplate` idea — a reusable string with named
`{placeholders}` — doesn't need the library, so it's implemented
directly in `prompt_templates.py`. If you have internet access, the
real equivalent is:

```python
pip install langchain-core
from langchain_core.prompts import PromptTemplate
template = PromptTemplate.from_template("Explain {topic} to a {audience}.")
template.format(topic="overfitting", audience="beginner")
```

which does exactly what `PromptTemplate.format()` below does.

## What's in this folder

- `ollama_client.py` — same client from W5D1, with one bug fixed (see below)
- `prompt_templates.py` — `PromptTemplate`, `FewShotPromptTemplate`, `chain_of_thought()`
- `system_prompt_experiment.py` — **Task A**: same question, 4 different system prompts (personas)
- `prompting_techniques.py` — **Task B**: zero-shot vs few-shot vs chain-of-thought on a sarcastic sentiment example
- `test_prompt_engineering.py` — 17 tests (templates, extraction helpers, end-to-end smoke tests against a fake server)
- `requirements.txt`
- `system_prompt_results.json`, `prompting_technique_results.json` — outputs _(generated when you run them)_

## Bug fix carried over from W5D1

`ollama_client.py`'s `chat()`/`list_models()` had `base_url: str =
DEFAULT_BASE_URL` as a default argument. In Python, that default is
bound once at import time, so reassigning the module-level
`DEFAULT_BASE_URL` later (e.g. to point at a test server) silently had
no effect on already-defined functions. Fixed by defaulting to `None`
and resolving `base_url or DEFAULT_BASE_URL` inside the function body,
so it's read fresh on every call. Caught this while writing the mock
server test for this task; backported the fix to W5D1's copy too.

## Setup

Same as W5D1 — Ollama installed, `llama3.2:3b` and `qwen2.5:3b` pulled,
`pip install -r requirements.txt`.

## Run

```
python -m unittest test_prompt_engineering -v   # 17 tests, mock server, no Ollama needed
python system_prompt_experiment.py              # Task A
python prompting_techniques.py                  # Task B
```

## Design decisions

**Task A — System prompt experiment.** Held the user question and
model completely fixed, and changed only the system prompt across 4
personas: a terse expert, a friendly teacher, a Socratic tutor who
only asks questions back, and a strict JSON-only responder. This
isolates exactly what a system prompt does and doesn't control: it
shapes tone, verbosity, format and stance, not facts the model doesn't
actually know.

**Task B — Prompting technique comparison.** Used a deliberately
sarcastic sentiment example ("Oh great, my flight got delayed... Love
this airline." → true label NEGATIVE) specifically because a model
that only pattern-matches surface-positive words ("great", "love")
will get it wrong — this is a real stress test, not a softball
example. Compared:

- **Zero-shot**: just ask
- **Few-shot**: 3 worked examples first, via `FewShotPromptTemplate`, including one other sarcastic example so the pattern to match is "sarcasm → negative," not just "negative words → negative"
- **Chain-of-thought**: append "Let's think step by step." (Kojima et al., 2022)

**Label extraction is a known-imperfect regex**, documented and tested
explicitly: if a response mentions both POSITIVE and NEGATIVE (e.g.
while reasoning through both options), only the _first_ occurrence is
extracted, which may not be the model's actual final answer. This is
called out in a dedicated test (`test_picks_first_match_when_both_present`)
rather than silently accepted.

## Testing

17 tests: template rendering (`PromptTemplate`, `FewShotPromptTemplate`,
`chain_of_thought`), the `extract_label` and `looks_like_json` helpers
(including their documented limitations), and end-to-end smoke tests
that run both scripts' full `main()` against a fake Ollama server. The
smoke tests verify the scripts produce well-formed output for every
persona/technique — they use a fake server that always replies
"NEGATIVE", so they check plumbing, not real model quality.

## Results

**Task A — same question, 4 system prompts:**

| Persona          | Word count | Notes                                                                                                                                                                                                                                 |
| ---------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| terse_expert     | 60         | Gave a direct, technical if/then answer (features → Logistic; complexity/non-linearity → Random Forest). Instructed to use "at most 2 sentences" — technically 2 sentences, but long ones; word limit not really respected in spirit. |
| friendly_teacher | 159        | Used a weather-forecasting analogy as instructed, warm tone, but got cut off mid-sentence at num_predict=200 before finishing the Random Forest half of the analogy.                                                                  |
| socratic_tutor   | 101        | Correctly gave zero direct recommendation — only 3 guiding questions about feature importance, linearity, and the actual goal (prediction vs. interpretability). Fully respected the "never give a direct answer" instruction.        |
| json_only        | 29         | Valid JSON: **True**. Returned `{"recommendation": "Random Forest", "reason": "..."}` with no markdown fences or extra text — fully respected the strict format instruction.                                                          |

**Task B — zero-shot vs few-shot vs CoT (sarcastic example, true label NEGATIVE):**

| Technique        | Predicted | Correct? |
| ---------------- | --------- | -------- |
| zero_shot        | NEGATIVE  | True     |
| few_shot         | NEGATIVE  | True     |
| chain_of_thought | NEGATIVE  | True     |

**What I observed:**

- System prompts controlled length (29 to 159 words), format (only `json_only` produced valid JSON), and stance (only `socratic_tutor` withheld a direct recommendation) — while the underlying facts and recommendation logic (features/linearity → Logistic; complexity/non-linearity → Random Forest) stayed consistent across the personas that did answer directly.
- The model didn't perfectly follow every instruction: `terse_expert`'s "2 sentences" constraint was technically met but not in spirit (60 words), and `friendly_teacher`'s response was truncated by the `num_predict=200` token limit before finishing its second analogy paragraph — worth raising `num_predict` for that persona if a complete analogy matters.
- On the sarcasm test, `llama3.2:3b` got it right with all three techniques (zero-shot, few-shot, chain-of-thought), so for this single example neither few-shot examples nor step-by-step reasoning were necessary to catch the sarcasm — the base model already handled it. This is one example with one seed, so it's not evidence the techniques are interchangeable in general, just that this particular case didn't need the extra scaffolding.

## Self-review checklist

- [x] Ran `system_prompt_experiment.py` and `prompting_techniques.py` on real Ollama
- [x] `python -m unittest test_prompt_engineering -v` passes (17 tests)
- [x] Results tables above filled with real numbers, not placeholders
- [x] Read the actual responses, not just the pass/fail label — the sarcasm
      example specifically needs a human read, since the extraction regex
      is a rough signal
- [x] Code commented; committed in 2+ commits; PR raised

## Author

Siriyala Nishar
