# Production ML Model Monitoring Strategy

## Scope
This project exposes a demo prediction endpoint. The included `/metrics` endpoint is
an in-memory teaching example, not a durable production monitoring backend. A real
deployment should export metrics to Prometheus/OpenTelemetry and visualize them in
Grafana or an equivalent platform.

## What to track

| Area | Metric / signal | Why it matters |
|---|---|---|
| Availability | `/health` success, uptime | Detect service outages |
| Traffic | prediction requests per minute | Detect traffic changes |
| Reliability | 4xx/5xx count and error rate | Detect malformed inputs and failures |
| Performance | p50/p95/p99 latency | Detect slow inference |
| Model quality | delayed ground-truth accuracy, precision/recall, MAE/RMSE as appropriate | Detect quality degradation |
| Data quality | missing values, schema violations, range checks | Detect invalid inputs |
| Data drift | feature distribution drift (PSI/KS or suitable tests) | Detect changes in incoming data |
| Model governance | model version, deployment timestamp | Trace prediction behavior |
| Resources | CPU, memory, container restarts | Detect capacity problems |

## Suggested alert policy

Thresholds below are starting points; calibrate them against a measured baseline.

- **Critical availability:** health check fails for 2 consecutive minutes.
- **High error rate:** 5xx rate exceeds 5% over a 5-minute window with sufficient traffic.
- **High latency:** p95 latency exceeds the service SLO for 10 minutes.
- **Data quality:** invalid-input rate rises above 2% over 10 minutes.
- **Drift warning:** a monitored feature drift statistic crosses its validated threshold.
- **Resource warning:** sustained CPU or memory above 85% for 10 minutes.

Alerts should include service name, model version, environment, dashboard link, and
a short runbook. Avoid paging on low-volume percentages without a minimum request count.

## Retraining and rollback triggers

1. Retrain only after confirming meaningful drift or sustained degradation using
   representative, validated ground-truth data.
2. Trigger a candidate training run when a quality metric breaches its agreed
   threshold for a defined evaluation window.
3. Compare the candidate with the currently deployed model on a fixed validation set
   and relevant slices. Require quality and fairness checks before promotion.
4. Deploy gradually (shadow/canary) and monitor the same metrics.
5. Roll back to the previous approved model if error rate, latency, or quality regresses.
6. Record dataset version, code revision, parameters, metrics, and model artifact
   in MLflow for reproducibility.

## Operational notes

- Keep application metrics, infrastructure metrics, and model-quality metrics distinct.
- Ground-truth labels may arrive late; do not treat unlabeled predictions as verified quality.
- Protect logs and monitoring data; do not log secrets or unnecessary personal data.
- The demo `/metrics` values reset on restart and are process-local. Use a durable
  metrics backend for multi-worker or production deployments.
