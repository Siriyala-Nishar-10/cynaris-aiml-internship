"""Compare flat vs router RAG and log each run to MLflow (if installed)."""
import csv, json, os, statistics as st

MODES = ["flat", "router"]
data, meta = {}, {}
for m in MODES:
    p = f"results/{m}_results.csv"
    if os.path.exists(p):
        data[m] = list(csv.DictReader(open(p, encoding="utf-8", newline="")))
        meta[m] = json.load(open(f"results/{m}_meta.json"))
if not data:
    raise SystemExit("No results yet. Run multi_doc_rag.py flat / router first.")


def rate(rows, col):
    vals = [r[col].strip() for r in rows if r[col].strip() in ("0", "1")]
    return (sum(map(int, vals)) / len(vals)) if vals else None


metrics = {}
for m, rows in data.items():
    lat = [float(r["total_s"]) for r in rows]
    metrics[m] = {"build_s": meta[m]["build_s"], "latency_mean_s": round(st.mean(lat), 2),
                  "latency_median_s": round(st.median(lat), 2), "errors": meta[m]["errors"],
                  "top_doc_accuracy": rate(rows, "top_doc_ok"), "keyword_accuracy": rate(rows, "keyword_hit"),
                  "manual_accuracy": rate(rows, "correct")}

print(f"{'metric':<22}" + "".join(f"{m:>12}" for m in data))
for k in ["build_s", "latency_mean_s", "latency_median_s", "errors", "top_doc_accuracy", "keyword_accuracy", "manual_accuracy"]:
    cells = []
    for m in data:
        v = metrics[m][k]
        cells.append("(unlabelled)" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)))
    print(f"{k:<22}" + "".join(f"{c:>12}" for c in cells))
print("\nkeyword_accuracy is an automatic first-pass check; manual_accuracy is your verified `correct` column.")

try:
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("w7d5-multidoc-rag")
    for m in data:
        with mlflow.start_run(run_name=m):
            mlflow.log_params({k: meta[m][k] for k in ("mode", "llm", "embed", "top_k", "documents")})
            mlflow.log_metrics({k: v for k, v in metrics[m].items() if v is not None})
            mlflow.log_artifact(f"results/{m}_results.csv")
    print("Logged to MLflow experiment 'w7d5-multidoc-rag' (run: mlflow ui --backend-store-uri sqlite:///mlflow.db)")
except ImportError:
    print("mlflow not installed - skipped logging")
