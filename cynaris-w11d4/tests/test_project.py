import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_ten_knowledge_base_records():
    rows=json.loads((ROOT/"data/knowledge_base.json").read_text(encoding="utf-8"))
    assert len(rows)==10 and all(x["content"] for x in rows)

def test_ten_question_answer_pairs():
    rows=json.loads((ROOT/"qa_pairs.json").read_text(encoding="utf-8"))
    assert len(rows)==10 and all(x["question"] and x["ground_truth"] for x in rows)

def test_four_ragas_metrics_are_configured():
    src=(ROOT/"evaluate_ragas.py").read_text(encoding="utf-8")
    for metric in ["faithfulness","answer_relevancy","context_precision","context_recall"]:
        assert metric in src

def test_two_retrieval_configurations_exist():
    src=(ROOT/"compare.py").read_text(encoding="utf-8")
    assert "chunk_size=450,k=2" in src
    assert "chunk_size=300,k=3" in src
