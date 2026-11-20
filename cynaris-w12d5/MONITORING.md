# Production Monitoring Strategy — W12D5

## Signals to track
- **Availability:** `/health` success rate and container restart count.
- **Traffic:** requests per minute and prediction volume.
- **Latency:** p50/p95/p99 prediction response time.
- **Errors:** HTTP 4xx/5xx rates, model-load failures and exceptions.
- **Model quality:** once labelled outcomes arrive, track MAE/RMSE and compare with a baseline.
- **Data quality and drift:** missing values, out-of-range features, feature distribution drift and prediction distribution drift.
- **Lineage:** model version, MLflow run ID, training data version and deployment timestamp.

The sample `/metrics` endpoint exposes in-process counters for demonstration. For production, export Prometheus metrics and use centralized logs and a durable monitoring backend.

## Example alert policy
| Signal | Example trigger | Response |
|---|---|---|
| Availability | health checks fail for 2 minutes | page owner; inspect container and dependencies |
| Error rate | 5xx > 2% for 5 minutes | investigate logs and consider rollback |
| Latency | p95 > 500 ms for 10 minutes | inspect load, resources and model path |
| Data drift | PSI > 0.2 on a key feature | investigate incoming data and segment impact |
| Model quality | MAE > 1.2x baseline on sufficient labelled data | review model and data; consider retraining |

Thresholds are illustrative and must be calibrated against real service traffic and business impact.

## Retraining decision
Do not retrain on drift alone. Trigger a review when meaningful drift persists, enough new labelled examples exist, or measured quality degrades beyond the agreed threshold. Validate data quality, compare candidate and champion models, record experiments in MLflow, run tests and bias/robustness checks, obtain approval, then deploy with a rollback path.

## Operational safeguards
- Version datasets and models; never overwrite the current production artifact silently.
- Keep secrets in GitHub Actions secrets, not source code.
- Use least-privilege registry permissions.
- Retain deployment and model lineage for auditability.
- The included synthetic dataset is for demonstration only, not a production training source.
