"""Ragas evaluation starter.

Ragas APIs evolve across releases. This module intentionally provides a small
dataset-preparation helper and a clearly marked integration point. Add metrics
and an LLM/embeddings provider after confirming the API for your installed
Ragas version.
"""
from datasets import Dataset


def build_evaluation_dataset(questions, answers, contexts, ground_truths):
    """Create the columns commonly used in Ragas evaluation datasets."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("All evaluation columns must have the same number of rows.")
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def evaluate_with_ragas(dataset):
    """TODO: configure Ragas metrics and providers for the chosen Ragas version."""
    raise NotImplementedError(
        "Configure Ragas metrics plus an LLM and embeddings provider, then call "
        "ragas.evaluate(...) using the API for your installed Ragas version."
    )
