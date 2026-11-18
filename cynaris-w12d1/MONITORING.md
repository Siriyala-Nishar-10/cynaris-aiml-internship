\# W12D1 — ML API Monitoring Strategy



\## 1. Metrics to Monitor



\- API request count

\- API response latency

\- HTTP error rate (4xx and 5xx)

\- Prediction response values

\- Model version

\- CPU and memory usage

\- Docker container health



\## 2. Alerts



| Alert | Condition | Action |

|---|---|---|

| High latency | Response time > 2 seconds | Investigate API |

| High error rate | Errors > 5% | Check application logs |

| Container failure | Health check fails | Restart and investigate |

| Resource usage | CPU or memory consistently high | Review resource allocation |

| Model performance | Prediction quality decreases | Evaluate model |



\## 3. Retraining Triggers



\- Significant data drift

\- Prediction accuracy degradation

\- Changes in input data distribution

\- New representative training data

\- Scheduled model evaluation



\## 4. Monitoring Tools



\- Docker logs

\- GitHub Actions

\- FastAPI health endpoint

\- MLflow experiment tracking



\## 5. Recovery Strategy



1\. Check application and container logs.

2\. Identify the source of failure.

3\. Roll back to the previous stable model if necessary.

4\. Run tests before redeployment.

5\. Monitor the API after recovery.



\## Conclusion



This monitoring strategy helps maintain API reliability, model quality, and deployment stability through continuous monitoring and controlled retraining.

