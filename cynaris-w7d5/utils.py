"""Pure helper functions (no LlamaIndex / Ollama needed) so they can be unit-tested."""
import re

# Which paper each of the 10 questions is about (2 questions per paper).
EXPECTED_DOC = {1: "growwise", 2: "growwise", 3: "mushroom", 4: "mushroom", 5: "banking",
                6: "banking", 7: "agentic", 8: "agentic", 9: "dashboard", 10: "dashboard"}

# Lower-case fragments that identify each paper from its file name.
DOC_KEYWORDS = {"growwise": ["growwise"], "mushroom": ["mushroom"],
                "banking": ["rag-based", "rag_based", "banking"],
                "agentic": ["agentic"], "dashboard": ["dashboard"]}

# One-line description of each paper; the router LLM reads these to pick a document.
DESCRIPTIONS = {
    "growwise": "GrowWise SME: an NLP chatbot (TF-IDF, cosine similarity) that recommends financial schemes and advice to small and medium enterprises.",
    "mushroom": "A CNN model that classifies 103 mushroom species from natural-habitat images, with data augmentation and accuracy results.",
    "banking": "A RAG framework for intelligent banking assistants: chunking, metadata enrichment, embeddings, and bgGPT answers over HR policy documents.",
    "agentic": "Agentic AI in finance: adoption surveys, use cases, PwC tax agents, risks, and human-in-the-loop integration.",
    "dashboard": "AI-powered business intelligence (Amazon QuickSight) used to reconcile financial data and surface hidden errors, plus analyst resistance.",
}

# Regex facts that a correct answer should contain. This is a quick first-pass
# check only; the `correct` column is still verified by hand against the paper.
ANSWER_KEY = {1: [r"87\.5"], 2: [r"tf-?\s?idf|cosine"], 3: [r"103", r"96\.7"],
              4: [r"augment|transformation"], 5: [r"semantic"], 6: [r"85"],
              7: [r"\b6\s?%|six percent"], 8: [r"one day|1 day"], 9: [r"203"],
              10: [r"aversion|identity|resist|trust"]}


def doc_key(filename: str):
    """Map a file name to one of the paper keys, or None if it is not recognised."""
    name = filename.lower()
    for key, fragments in DOC_KEYWORDS.items():
        if any(f in name for f in fragments):
            return key
    return None


def keyword_hit(q_id: int, answer: str) -> bool:
    """True if the answer contains every fact in the answer key for this question."""
    return all(re.search(p, answer, re.I) for p in ANSWER_KEY[q_id])
