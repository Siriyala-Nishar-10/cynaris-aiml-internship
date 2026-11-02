"""Ragas layer: score faithfulness and answer relevancy with the local models."""
from __future__ import annotations

from .config import Settings


def score(question: str, answer: str, contexts: list[str], cfg: Settings) -> dict[str, float]:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    dataset = EvaluationDataset.from_list([{
        "user_input": question,
        "response": answer,
        "retrieved_contexts": contexts,
    }])
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(ChatOllama(model=cfg.chat_model, base_url=cfg.ollama_url)),
        embeddings=LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=cfg.embed_model, base_url=cfg.ollama_url)),
    )
    row = result.to_pandas().iloc[0]
    return {"faithfulness": float(row["faithfulness"]),
            "answer_relevancy": float(row["answer_relevancy"])}
