import json
from pathlib import Path

import mlflow
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from .config import MLFLOW_TRACKING_URI
from .rag_pipeline import TrackedRAG


def run_evaluation():
    rag = TrackedRAG()
    qa = json.loads(Path("data/evaluation_questions.json").read_text(encoding="utf-8"))
    rows = []
    for item in qa:
        output = rag.ask(item["question"])
        rows.append({
            "question": item["question"],
            "answer": output["answer"],
            "contexts": output["contexts"],
            "ground_truth": item["ground_truth"],
        })

    dataset = Dataset.from_list(rows)
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    frame = scores.to_pandas()
    Path("results").mkdir(exist_ok=True)
    output_path = "results/ragas_scores.csv"
    frame.to_csv(output_path, index=False)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("W11D5-Tracked-Evaluated-RAG")
    with mlflow.start_run(run_name="ragas-evaluation"):
        mlflow.log_param("evaluation_questions", len(rows))
        mlflow.log_artifact(output_path)
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if metric in frame.columns:
                value = pd.to_numeric(frame[metric], errors="coerce").mean()
                if pd.notna(value):
                    mlflow.log_metric(metric, float(value))
    print(frame)
    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    run_evaluation()
