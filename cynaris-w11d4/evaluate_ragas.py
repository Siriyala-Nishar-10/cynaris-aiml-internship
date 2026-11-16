
import os
import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama, OllamaEmbeddings


# Local Ollama configuration
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    num_predict=256,
    request_timeout=300,
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]


def evaluate_rows(rows, experiment_name="baseline"):

    if not rows:
        raise ValueError("No evaluation rows provided.")

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for row in rows:
        data["question"].append(row["question"])
        data["answer"].append(row["answer"])

        contexts = row.get("contexts", row.get("context", []))

        if isinstance(contexts, str):
            contexts = [contexts]

        data["contexts"].append(contexts)

        ground_truth = row.get(
            "ground_truth",
            row.get("reference", "")
        )

        data["ground_truth"].append(ground_truth)

    dataset = Dataset.from_dict(data)

    print(f"\nStarting Ragas evaluation: {experiment_name}")

    run_config = RunConfig(
        timeout=300,
        max_retries=2,
        max_wait=30,
        max_workers=1,
    )

    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )

    scores = result.to_pandas()

    os.makedirs("results", exist_ok=True)

    output_file = (
        f"results/{experiment_name}_ragas_scores.csv"
    )

    scores.to_csv(output_file, index=False)

    summary = {}

    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]:

        if metric in scores.columns:
            valid_scores = pd.to_numeric(
                scores[metric],
                errors="coerce"
            ).dropna()

            summary[metric] = (
                round(float(valid_scores.mean()), 4)
                if not valid_scores.empty
                else None
            )

    print(f"\n{experiment_name.upper()} RESULTS")

    for metric, score in summary.items():
        print(f"{metric}: {score}")

    print(f"\nResults saved to: {output_file}")

    return summary


if __name__ == "__main__":
    print("W11D4 — Ragas Evaluation")
    print("Use compare.py to execute baseline and optimized evaluations.")
