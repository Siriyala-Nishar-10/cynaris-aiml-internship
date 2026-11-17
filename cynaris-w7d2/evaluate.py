"""Precision@3 for every mode whose results CSV has hand labels (bm25, dense, hybrid)."""
import csv, os, sys
from collections import defaultdict

MODES = ["bm25", "dense", "hybrid"]


def precision(path):
    per_q = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["relevant"].strip() in ("0", "1"):
                per_q[int(row["q_id"])].append(int(row["relevant"]))
    return {q: sum(v) / len(v) for q, v in per_q.items()}


scores = {}
for m in MODES:
    path = f"results/{m}_results.csv"
    if os.path.exists(path):
        p = precision(path)
        if p:
            scores[m] = p
        else:
            print(f"(skipping {m}: no labels in the 'relevant' column yet)")
if not scores:
    sys.exit("No labelled results found. Fill the 'relevant' column (1/0) first.")

modes = list(scores)
print(f"{'Q':>3} | " + " | ".join(f"{m + ' P@3':>11}" for m in modes))
for q in sorted(set().union(*[set(s) for s in scores.values()])):
    print(f"{q:>3} | " + " | ".join(f"{scores[m].get(q, float('nan')):>11.2f}" for m in modes))
print("-" * (6 + 14 * len(modes)))
print(f"Avg | " + " | ".join(f"{sum(scores[m].values()) / len(scores[m]):>11.2f}" for m in modes))
