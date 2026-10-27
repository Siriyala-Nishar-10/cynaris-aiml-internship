"""Compare the in-memory index and ChromaDB: index build time, latency, and answer accuracy."""
import csv, json, os, statistics as st

MODES = ["memory", "chroma"]
data = {}
for m in MODES:
    p = f"results/{m}_results.csv"
    if os.path.exists(p):
        data[m] = list(csv.DictReader(open(p, encoding="utf-8", newline="")))
if not data:
    raise SystemExit("No results yet. Run rag_compare.py memory / chroma first.")

print(f"{'metric':<28}" + "".join(f"{m:>12}" for m in data))
def row(label, fn):
    print(f"{label:<28}" + "".join(f"{fn(m):>12}" for m in data))

row("index build (s)", lambda m: json.load(open(f"results/{m}_meta.json"))["build_s"])
row("retrieval mean (s)", lambda m: f"{st.mean(float(r['retrieval_s']) for r in data[m]):.3f}")
row("retrieval median (s)", lambda m: f"{st.median(float(r['retrieval_s']) for r in data[m]):.3f}")
row("total query mean (s)", lambda m: f"{st.mean(float(r['total_s']) for r in data[m]):.2f}")
row("total query median (s)", lambda m: f"{st.median(float(r['total_s']) for r in data[m]):.2f}")

def acc(m):
    labels = [r["correct"].strip() for r in data[m] if r["correct"].strip() in ("0", "1")]
    return f"{sum(map(int, labels))}/{len(labels)}" if labels else "(unlabelled)"
row("correct answers", acc)
print("\nretrieval_s = query embedding + vector search; total_s adds LLM generation (usually most of the time).")
