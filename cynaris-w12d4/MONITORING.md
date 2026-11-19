# Production Monitoring Strategy

## Signals to track
- **Availability:** `/health` response and HTTP 5xx rate.
- **Traffic:** prediction requests per minute and request volume changes.
- **Latency:** average and p95 prediction latency from `/metrics`.
- **Errors:** invalid requests, exceptions, timeout rate, and failed predictions.
- **Model quality:** once labelled outcomes arrive, track MAE/RMSE (regression) or precision/recall (classification).
- **Data quality/drift:** missing values, schema violations, feature ranges, and distribution drift.
- **Model lineage:** model version, deployment commit, MLflow run/model URI.

## Suggested alert rules (initial thresholds; tune to baseline)
| Signal | Alert condition | First response |
|---|---|---|
| Availability | Health check fails 3 consecutive times | Check container, logs, and dependencies |
| Server errors | 5xx rate > 2% for 5 minutes | Inspect recent deploy and error logs |
| Latency | p95 > 500 ms for 5 minutes | Check resource use and slow dependencies |
| Traffic | Sudden sustained drop/spike vs normal baseline | Check upstream caller and service health |
| Data quality | Invalid/schema failures > 1% | Quarantine bad batch and inspect source |
| Drift | PSI > 0.2 or approved drift test threshold | Investigate; compare with labelled performance |

## Retraining triggers
1. Confirmed feature drift over the agreed threshold for a sustained period.
2. Labelled model quality falls below the business acceptance threshold.
3. New representative data is available and passes data-quality checks.
4. A scheduled review is due and evaluation shows a candidate model improves on the champion.

Never retrain automatically from drift alone. Validate data, train a candidate, compare it against the current model, record the experiment in MLflow, run evaluation and (where applicable) Ragas checks for retrieval-grounded components, then require approval before promotion. Keep rollback to the previous model version available.

## Operational notes
- This project exposes lightweight in-process demo metrics at `/metrics`; counters reset when the process restarts and are not a production metrics backend.
- In production, export metrics to Prometheus/OpenTelemetry and dashboards/alerts to the team's monitoring platform.
- Keep logs structured, avoid logging secrets or raw sensitive feature values, and attach model version plus deployment commit to each event.
