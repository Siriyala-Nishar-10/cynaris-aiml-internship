"""W8D2 - Evaluate the RAG pipeline with Ragas (local Ollama judge).

Usage:
    python evaluate_ragas.py --label baseline --chunk 1000 --k 3
    python evaluate_ragas.py --label optimised --chunk 1000 --k 5
    python evaluate_ragas.py --limit 2        # quick smoke test on 2 questions

Writes results/ragas_<label>.csv (per-question scores) and results/summary_<label>.json.
The judge model defaults to LLM_MODEL; override with  $env:JUDGE_MODEL="qwen2.5:3b".
A local 3B judge is slow and sometimes returns unparseable output; those scores
come back as NaN and are excluded from the means (see valid_counts in the summary).
"""
import argparse
import json
import os
import time

import compat

compat.apply()   # must run before ragas is imported (see compat.py)

import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.run_config import RunConfig

try:
    from ragas.metrics import ResponseRelevancy
except ImportError:                                   # older Ragas versions
    from ragas.metrics import AnswerRelevancy as ResponseRelevancy

from rag_pipeline import EMBED_MODEL, LLM_MODEL, OLLAMA_HOST, RagPipeline
from utils import METRICS, load_qa, lowest_metric, pick_columns, summarise

JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)

ap = argparse.ArgumentParser()
ap.add_argument("--label", default="baseline")
ap.add_argument("--chunk", type=int, default=1000, help="chunk size in characters")
ap.add_argument("--k", type=int, default=3, help="chunks retrieved per question")
ap.add_argument("--limit", type=int, default=0, help="only use the first N questions (0 = all)")
args = ap.parse_args()

qa = load_qa()
if args.limit:
    qa = qa[:args.limit]

print(f"[{args.label}] building RAG: chunk_size={args.chunk}, k={args.k}")
rag = RagPipeline(args.chunk, args.k)
print(f"   {rag.n_chunks} chunks indexed")

samples, answers = [], []
t0 = time.perf_counter()
for item in qa:
    answer, contexts = rag.ask(item["question"])
    answers.append(answer)
    samples.append(SingleTurnSample(user_input=item["question"], response=answer,
                                    retrieved_contexts=contexts, reference=item["reference"]))
    print(f"Q{item['id']}: {answer[:110]}")
rag_s = time.perf_counter() - t0

print(f"\nRunning Ragas with judge {JUDGE_MODEL} - this is slow on CPU (minutes).")
judge = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_HOST, temperature=0))
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST))
metrics = [Faithfulness(llm=judge),
           ResponseRelevancy(llm=judge, embeddings=embeddings),
           LLMContextPrecisionWithReference(llm=judge),
           LLMContextRecall(llm=judge)]
result = evaluate(dataset=EvaluationDataset(samples=samples), metrics=metrics,
                  run_config=RunConfig(timeout=300, max_workers=1), raise_exceptions=False)
df = result.to_pandas()
cols = pick_columns(list(df.columns))

rows = []
for i, item in enumerate(qa):
    row = {m: (float(df.iloc[i][cols[m]]) if m in cols else None) for m in METRICS}
    rows.append({"q_id": item["id"], "question": item["question"], "answer": answers[i], **row})
os.makedirs("results", exist_ok=True)
pd.DataFrame(rows).to_csv(f"results/ragas_{args.label}.csv", index=False)

summary = summarise(rows)
summary.update({"label": args.label, "chunk_size": args.chunk, "k": args.k, "chunks": rag.n_chunks,
                "llm": LLM_MODEL, "judge": JUDGE_MODEL, "embed": EMBED_MODEL,
                "rag_answer_seconds": round(rag_s, 1), "lowest_metric": lowest_metric(summary["means"])})
with open(f"results/summary_{args.label}.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nMean scores (higher is better):")
for m in METRICS:
    print(f"  {m:<18} {summary['means'][m]}   (valid {summary['valid_counts'][m]}/{summary['n_questions']})")
print(f"Lowest metric: {summary['lowest_metric']}")

try:
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("w8d2-ragas")
    with mlflow.start_run(run_name=args.label):
        mlflow.log_params({k: summary[k] for k in ("chunk_size", "k", "llm", "judge", "embed", "chunks")})
        mlflow.log_metrics({m: v for m, v in summary["means"].items() if v is not None})
        mlflow.log_artifact(f"results/ragas_{args.label}.csv")
    print("Logged to MLflow experiment 'w8d2-ragas'")
except ImportError:
    pass