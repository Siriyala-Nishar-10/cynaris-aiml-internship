# Routing Strategy: Local Q&A Bot (Ollama + ChromaDB)

**Project:** local document Q&A bot (W5D5), `llama3.2:3b` and `qwen2.5:3b` through Ollama, ChromaDB for retrieval.
**Question:** which queries stay on the local model, and which should go to a paid API?

## 1. Goal and constraints

- Keep the cost of the bot as low as possible without a drop in answer quality that users would notice.
- Documents may be private, so some queries must **never** leave the machine, whatever their difficulty.
- Local inference is free per call but slow on this hardware: **5 to 45 seconds per answer** on CPU in the W5D5 runs. Speed is part of the decision, not just cost.

## 2. What the W5D5 experiments showed

| Observation | Source | Consequence for routing |
|---|---|---|
| Both 3B models answered in-document factual questions correctly and almost identically | `compare_models.py` | Simple lookups are safe to keep local |
| Retrieval similarity separated real questions (0.49 to 0.77) from an out-of-scope one (about 0.40) | `run_questions.py` | Top similarity is a cheap, useful routing signal |
| `llama3.2:3b` cited a page that was not in the context | W5D5 README | Local answers need a citation check; do not trust them on high-stakes questions |
| `qwen2.5:3b` followed the exact-refusal rule, `llama3.2:3b` added commentary | `compare_models.py` | The local tier should use the model that follows instructions best |
| Both models missed the "Bulgarian" part of a two-part question | W5D5 README | Multi-part questions are a weak spot for a 3B model |

## 3. The tiers

| Tier | Handler | Cost | Handles |
|---|---|---|---|
| 0 | **Semantic cache** | about zero | Questions similar (cosine ≥ the threshold chosen in the cache notebook) to one already answered |
| 1 | **Local model** (`qwen2.5:3b` or `llama3.2:3b`) | ₹0 per call | Single-fact, in-document lookups; short answers; any query on private documents |
| 2 | **Cheap API** (GPT-4o mini) | about ₹0.016 per typical call* | Summaries across several chunks, multi-part questions, and queries where local latency is too high |
| 3 | **Strong API** (GPT-4o) | about ₹0.46 per typical call* | Complex reasoning, comparisons across documents, long contexts, and escalations from Tier 2 |

\*600 input and 150 output tokens at the W5D6 lesson prices and ₹88 per USD. See the audit spreadsheet for the full calculation.

**Rule above all tiers: sensitive documents are never sent to an API.** If a query on a sensitive collection is too hard for the local model, the bot answers locally with a low-confidence warning or declines, instead of escalating.

## 4. Routing signals

Cheap signals are computed before any LLM call, so routing itself costs almost nothing.

| Signal | How it is measured | Points toward |
|---|---|---|
| Cache similarity | cosine to the nearest cached question | ≥ threshold: Tier 0 |
| Collection sensitivity | metadata flag on the document | `sensitive`: Tier 1 only |
| Top retrieval similarity | best chunk score from ChromaDB | < 0.40: refuse (nothing relevant); ≥ 0.50: Tier 1 is likely enough |
| Question complexity | length, number of question marks or "and", words like *compare*, *why*, *summarise*, *explain differences* | high: Tier 2 or 3 |
| Context size | tokens in the retrieved prompt (`tiktoken`) | above the local context budget: Tier 2 or 3 |
| Latency budget | caller says the answer is needed interactively | tight: API tiers |
| Confidence of the previous tier | citation check passed? answer says "I don't know"? | failure: escalate one tier |

## 5. Decision flow

```python
def route(question, hits, collection_sensitive, latency_budget_s=None):
    # Tier 0: semantic cache
    cached, sim = cache.get(question)
    if cached:
        return "cache"

    top_sim = hits[0]["sim"]
    if top_sim < 0.40:
        return "refuse"                      # nothing relevant in the documents

    complex_q = is_complex(question)         # length, "compare/why/summarise", multi-part
    big_context = count_tokens(hits) > LOCAL_CONTEXT_BUDGET

    if collection_sensitive:
        return "local"                       # privacy rule: never leaves the machine

    if complex_q and big_context:
        return "strong_api"                  # Tier 3
    if complex_q or big_context:
        return "cheap_api"                   # Tier 2
    if latency_budget_s is not None and latency_budget_s < 10:
        return "cheap_api"                   # local CPU is too slow for this deadline
    return "local"                           # Tier 1: simple lookup with good retrieval

# Escalation: if the chosen tier's answer fails a check, move up one tier
#   local      -> cheap_api   (citation not in retrieved context, or "I don't know" with top_sim >= 0.50)
#   cheap_api  -> strong_api  (self-reported low confidence or failed validation)
# Every answer is stored in the cache, so a repeated question costs nothing.
```

## 6. Expected cost impact

The W5D6 lesson reports that about **70%** of queries can be handled by the cheap model. Using the audit spreadsheet's structure (compression keeps about 40 to 45% of input tokens, a cache hit rate set from the cache notebook, and 70% of the remaining traffic on the cheap tier), the optimised cost per query comes out far below sending everything to GPT-4o. This bot's own advantage is Tier 1: queries that stay local cost ₹0, so the real saving can be higher than the lesson's 60 to 80% range.

These figures depend on assumptions (hit rate, routing share, traffic) that must be replaced with measured values before they are quoted. The spreadsheet marks each one.

## 7. Validation plan

1. **Label a test set:** about 50 questions marked simple, medium or complex, plus a few out-of-scope ones. Reuse the questions from `run_questions.py` and extend them.
2. **Measure routing accuracy:** what share of questions each rule sends to the right tier.
3. **Measure answer quality per tier:** keyword recall (as in `compress_prompt.py`) plus a manual check against the source pages.
4. **Track real spend:** `cost_tracker.py` logs tokens and INR per request, so the daily total and the share of traffic per tier can be compared with the plan.
5. **Tune thresholds:** the 0.40 refusal threshold and the cache threshold are specific to the `nomic-embed-text` embeddings and this document, so re-tune them if either changes.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Wrong cache hit returns an answer to a different question | Pick the lowest threshold with **zero wrong hits** in the cache notebook's sweep; add a TTL; keep context-dependent questions out of the cache |
| Small local model gives a confident wrong answer | Citation check on every local answer; escalate on failure |
| Routing heuristics misjudge complexity | Log every decision; review misrouted cases; replace rules with a small classifier once enough labelled data exists |
| API prices or the exchange rate change | Prices and USD to INR live in the Assumptions sheet, not in code |
| Latency spikes on the local tier | Latency budget signal; send time-critical queries to the cheap API |
