import json
from pathlib import Path


def test_knowledge_base_has_records():
    records = json.loads(Path("data/knowledge_base.json").read_text(encoding="utf-8"))
    assert len(records) >= 3
    assert all({"id", "topic", "content"} <= set(row) for row in records)


def test_evaluation_questions_have_ground_truth():
    rows = json.loads(Path("data/evaluation_questions.json").read_text(encoding="utf-8"))
    assert len(rows) >= 3
    assert all(row.get("question") and row.get("ground_truth") for row in rows)
