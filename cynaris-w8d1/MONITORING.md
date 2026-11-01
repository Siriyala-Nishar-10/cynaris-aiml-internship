# Monitoring Strategy — RAG API

Starting thresholds below come from my own runs (Week 7: 10-question golden set,
local Ollama on CPU). They are starting values to tune once real traffic exists.

## 1. What to track

| Area | Metric | How it is measured | Baseline |
| ---- | ------ | ------------------ | -------- |
| Service | request rate, error rate | `/stats` counters and `query_failed` log lines | 0 errors in 10 test queries |
| Service | latency p50 / p95 | `latency_s` in each `query_ok` log line and API response | median 10–45 s on CPU with a 3B model |
| Service | health and restarts | `/health` plus Docker `HEALTHCHECK`, container restart count | healthy |
| Service | CPU, memory | container metrics | measure after first deploy |
| Retrieval | right-document rate | nightly golden set (`top_doc_ok`) | 10/10 |
| Retrieval | similarity score of top chunk | log the top score per query | record the median after deploy |
| Answer quality | golden-set accuracy | nightly run of the 10 labelled questions | flat 8/10, router 9/10 |
| Answer quality | faithfulness / answer relevance | Ragas (set up in W8D2) | to be measured |
| Answer quality | empty or "I don't know" answers | share of answers with no content | 0% on the golden set |
| Data | corpus changes | count of new or edited documents, index age | index built at deploy |
| Users | thumbs up/down or corrections | feedback endpoint (future) | – |

## 2. Alerts

| Alert | Condition | Severity | First action |
| ----- | --------- | -------- | ------------ |
| API down | `/health` fails 3 checks in a row | critical | restart container, check Ollama is reachable |
| High error rate | 5xx rate > 5% over 15 min (> 20% = critical) | warning / critical | read `query_failed` logs, check the Ollama connection |
| Slow answers | p95 latency > 2x the measured baseline for 15 min | warning | check Ollama model loading and CPU load |
| Quality drop | nightly golden-set accuracy below 7/10 | warning | compare with the last good run in MLflow |
| Retrieval drift | right-document rate below 9/10 | warning | inspect chunking and the embedding model |
| Weak answers | empty answers > 10% of queries | warning | check for missing documents or a changed prompt |
| Faithfulness | Ragas faithfulness below 0.7 | warning | review retrieval and prompt |
| Stale index | index older than 30 days with new documents available | info | schedule a re-index |

## 3. What triggers a re-index or retrain

RAG has no training loop in this project, so "retrain" means re-index, re-tune
or swap a model. Triggers:

| Trigger | Action |
| ------- | ------ |
| 10% or more of documents are new or changed, or any document is replaced | re-index (re-chunk and re-embed) |
| Golden-set accuracy below 7/10 on two nightly runs in a row | investigate chunk size, top-k, or hybrid retrieval (W7 finding: answers split across chunks) |
| Embedding model changes | full re-embed of every document; never mix embeddings from two models |
| LLM model changes | re-run the golden set and compare with the baseline before releasing |
| Ragas faithfulness below 0.7 for a week | review the prompt and retrieved context |
| More than 20% negative user feedback in a week | add the failing questions to the golden set, then fix retrieval |

## 4. Process

- Run the golden set every night (reusing `summarise.py`) and log accuracy and
  latency to MLflow, so quality changes show up as a trend.
- Keep structured JSON-style log lines (`query_ok`, `query_failed`) so a log
  tool can calculate error rate and latency percentiles.
- Image tags use the commit SHA, so a bad release can be rolled back by
  redeploying the previous tag.
- Send critical alerts to one channel that the team watches; review warnings
  weekly.
