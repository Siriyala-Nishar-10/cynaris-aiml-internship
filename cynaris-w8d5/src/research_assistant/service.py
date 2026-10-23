"""One shared entry point for answering a question (used by the CLI and the web UI).
Every call is a tracked MLflow run."""
from __future__ import annotations

import time

from .config import Settings
from .memory import VectorMemory


def answer_question(question: str, cfg: Settings, memory: VectorMemory,
                    evaluate: bool = False) -> dict:
    """Run the LangGraph pipeline and return answer, sources, latency and scores."""
    import mlflow
    from .evaluate import score
    from .graph import build_graph

    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.experiment)
    graph = build_graph(cfg, memory)

    with mlflow.start_run():
        mlflow.log_params({"chat_model": cfg.chat_model, "embed_model": cfg.embed_model,
                           "top_k": cfg.top_k, "chunk_size": cfg.chunk_size,
                           "chunk_overlap": cfg.chunk_overlap})
        start = time.perf_counter()
        state = graph.invoke({"question": question})
        latency = time.perf_counter() - start
        contexts = state.get("contexts", [])
        mlflow.log_metric("latency_s", latency)
        mlflow.log_metric("num_contexts", len(contexts))
        mlflow.log_text(question, "question.txt")
        mlflow.log_text(state["answer"], "answer.txt")
        scores: dict[str, float] = {}
        if evaluate and contexts:
            scores = score(question, state["answer"], contexts, cfg)
            mlflow.log_metrics(scores)

    return {
        "answer": state["answer"],
        "sources": [{"file": f, "text": t}
                    for f, t in zip(state.get("sources", []), contexts)],
        "latency_s": round(latency, 2),
        "scores": scores,
    }
