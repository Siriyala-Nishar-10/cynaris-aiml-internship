"""Compute precision@3 for BM25 vs Dense from the hand-labelled CSVs."""
import csv, sys
from collections import defaultdict


def precision(path):
    per_q = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["relevant"].strip() in ("0", "1"):
                per_q[int(row["q_id"])].append(int(row["relevant"]))
    if not per_q:
        sys.exit(f"{path}: no labels found. Fill the 'relevant' column with 1 or 0.")
    return {q: sum(v) / len(v) for q, v in per_q.items()}


bm25, dense = precision("results/bm25_results.csv"), precision("results/dense_results.csv")
print(f"{'Q':>3} | {'BM25 P@3':>9} | {'Dense P@3':>9}")
for q in sorted(set(bm25) | set(dense)):
    print(f"{q:>3} | {bm25.get(q, float('nan')):>9.2f} | {dense.get(q, float('nan')):>9.2f}")
print("-" * 30)
print(f"Avg | {sum(bm25.values())/len(bm25):>9.2f} | {sum(dense.values())/len(dense):>9.2f}")
