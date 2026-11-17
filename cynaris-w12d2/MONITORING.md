# ML API Monitoring Strategy

## Metrics to track
- API request count, latency (p50/p95/p99), and error rate by endpoint.
- Container CPU, memory, restart count, and health-check status.
- Prediction volume and response distribution.
- Model version on every prediction.
- Input feature schema failures and missing/invalid values.
- Model quality metrics when delayed ground-truth labels become available.

## Alerts
- `/health` fails for 2 consecutive checks.
- HTTP 5xx rate exceeds 5% over a 5-minute window.
- p95 prediction latency exceeds the agreed service objective for 5 minutes.
- Container restarts repeatedly or memory/CPU remains near its configured limit.
- Input validation failures rise sharply.
- Model-quality metric drops below the team's approved threshold.

## Retraining triggers
- Verified model-quality degradation on newly labelled data.
- Sustained feature or prediction drift confirmed against a baseline.
- A scheduled review interval is reached and sufficient new labelled data exists.
- A material change in the input schema or business requirements occurs.

Retraining must be evaluated offline, compared with the current model, approved,
and deployed with a recorded model version and rollback option. Do not retrain
automatically from a single alert or an unverified drift signal.

## Operational notes
This project is a demonstration API. Production monitoring should export metrics
to an observability platform and define thresholds from measured baselines.
