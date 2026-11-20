from src.evaluation import build_evaluation_dataset
import pytest


def test_evaluation_dataset_columns():
    dataset = build_evaluation_dataset(
        questions=["What is LangGraph?"],
        answers=["A graph-based orchestration framework."],
        contexts=[["Reference context."]],
        ground_truths=["A framework for stateful graph workflows."],
    )
    assert set(dataset.column_names) == {
        "question", "answer", "contexts", "ground_truth"
    }
    assert len(dataset) == 1


def test_evaluation_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        build_evaluation_dataset(
            questions=["Q1", "Q2"],
            answers=["A1"],
            contexts=[["C1"]],
            ground_truths=["G1"],
        )
