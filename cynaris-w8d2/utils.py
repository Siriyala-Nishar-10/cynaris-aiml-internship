"""Pure helpers (no Ollama / Ragas needed) so they can be unit-tested."""
import json
import math

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_qa(path="qa_pairs.json"):
    """Load the question / reference-answer pairs and check each one is complete."""
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    for p in pairs:
        if not p.get("question") or not p.get("reference"):
            raise ValueError(f"Incomplete Q&A pair: {p}")
    return pairs


def pick_columns(columns):
    """Map Ragas output column names (they differ between versions) to our 4 metric names."""
    found = {}
    for col in columns:
        name = col.lower()
        if "faith" in name:
            found["faithfulness"] = col
        elif "relevan" in name:
            found["answer_relevancy"] = col
        elif "precision" in name:
            found["context_precision"] = col
        elif "recall" in name:
            found["context_recall"] = col
    return found


def _valid(values):
    return [v for v in values if v is not None and not math.isnan(v)]


def summarise(rows):
    """rows: list of {metric: score}. Returns mean per metric (NaN judge failures ignored) and counts."""
    means, valid = {}, {}
    for m in METRICS:
        vals = _valid([r.get(m) for r in rows])
        means[m] = round(sum(vals) / len(vals), 3) if vals else None
        valid[m] = len(vals)
    return {"means": means, "valid_counts": valid, "n_questions": len(rows)}


def lowest_metric(means):
    """Name of the metric with the lowest mean score, ignoring metrics that have no valid score."""
    scored = {m: v for m, v in means.items() if v is not None}
    return min(scored, key=scored.get) if scored else None
