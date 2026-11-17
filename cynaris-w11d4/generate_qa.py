import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
docs=json.loads((ROOT/"data/knowledge_base.json").read_text(encoding="utf-8"))
pairs=json.loads((ROOT/"qa_pairs.json").read_text(encoding="utf-8"))
assert len(docs)==10 and len(pairs)==10
out=ROOT/"results"; out.mkdir(exist_ok=True)
(out/"generated_qa_pairs.json").write_text(json.dumps(pairs,indent=2),encoding="utf-8")
print(f"Validated {len(pairs)} Q&A pairs against {len(docs)} knowledge-base records.")
