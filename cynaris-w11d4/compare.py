import json
from pathlib import Path
import pandas as pd
from rag_pipeline import run_rag
from evaluate_ragas import evaluate_rows

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/"results"

def run_experiment():
    qa=json.loads((ROOT/"qa_pairs.json").read_text(encoding="utf-8"))
    print("\\n=== BASELINE: chunk_size=450, retrieval_k=2 ===")
    baseline_rows=run_rag(qa,chunk_size=450,k=2)
    (RESULTS/"baseline_answers.json").write_text(json.dumps(baseline_rows,indent=2),encoding="utf-8")
    baseline=evaluate_rows(baseline_rows,"baseline")
    print("\\n=== OPTIMIZED: chunk_size=300, retrieval_k=3 ===")
    tuned_rows=run_rag(qa,chunk_size=300,k=3)
    (RESULTS/"optimized_answers.json").write_text(json.dumps(tuned_rows,indent=2),encoding="utf-8")
    tuned=evaluate_rows(tuned_rows,"optimized")
    comparison=pd.DataFrame([{"configuration":"baseline",**baseline},{"configuration":"optimized",**tuned}])
    comparison.to_csv(RESULTS/"metric_comparison.csv",index=False)
    print("\\n=== METRIC COMPARISON ===")
    print(comparison.to_string(index=False))
    lowest=min(baseline,key=baseline.get)
    print(f"\\nLowest baseline metric: {lowest}")
    print(f"Change after tuning: {tuned[lowest]-baseline[lowest]:+.4f}")
    print("Saved outputs in results/")

if __name__=="__main__": run_experiment()
