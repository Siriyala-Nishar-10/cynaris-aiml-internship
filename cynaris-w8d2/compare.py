"""Print baseline vs optimised Ragas scores and the change in each metric.

Usage:  python compare.py baseline optimised
"""
import json
import sys

from utils import METRICS

labels = sys.argv[1:] or ["baseline", "optimised"]
data = {lab: json.load(open(f"results/summary_{lab}.json")) for lab in labels}

print(f"{'metric':<20}" + "".join(f"{lab:>12}" for lab in labels) + ("      change" if len(labels) == 2 else ""))
for m in METRICS:
    vals = [data[lab]["means"][m] for lab in labels]
    cells = "".join(f"{'n/a' if v is None else f'{v:.3f}':>12}" for v in vals)
    change = ""
    if len(vals) == 2 and None not in vals:
        change = f"{vals[1] - vals[0]:+12.3f}"
    print(f"{m:<20}{cells}{change}")
print()
for lab in labels:
    d = data[lab]
    print(f"{lab}: chunk_size={d['chunk_size']}, k={d['k']}, lowest metric = {d['lowest_metric']}")
