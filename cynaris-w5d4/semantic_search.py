"""
W5D4: Semantic search with ChromaDB (semantic vs keyword, top-k, filters)
--------------------------------------------------------------------------
Builds on W5D3. Reuses the same 20 documents and the same embedding model,
then shows what makes semantic search different from keyword search:

  1. Semantic vs keyword: a query that shares NO words with the best document
     still finds it, because embeddings match meaning, not spelling.
  2. Top-k and a similarity threshold: return only results that are close
     enough, instead of always returning k results.
  3. Combined metadata filter: restrict the search to several categories
     with `$in`, and combine conditions with `$and`.

Chroma returns cosine DISTANCE, so similarity = 1 - distance.

Run:  python semantic_search.py
      python semantic_search.py "your own query"
Needs: ingest_docs.py in the same folder (for DOCUMENTS and embed),
       pip install chromadb ollama   |   ollama pull nomic-embed-text
"""

import re
import sys

import chromadb

from ingest_docs import DB_PATH, DOCUMENTS, embed

COLLECTION = "semantic_docs"
MIN_SIM = 0.50   # threshold: results below this similarity are dropped
STOPWORDS = {"a", "an", "the", "of", "to", "in", "on", "and", "or", "is", "are",
             "how", "do", "does", "what", "for", "from", "by", "with", "it", "its"}

QUERIES = [
    "How does a machine improve itself over time?",   # no word overlap with the best doc
    "Where can I find the tallest peak?",
    "Tools for shipping software in containers",
]


def build_collection():
    """Create (or refresh) a cosine collection with the 20 documents."""
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    texts = [t for t, _ in DOCUMENTS]
    col.upsert(
        ids=[f"doc{i}" for i in range(len(DOCUMENTS))],
        documents=texts,
        embeddings=embed(texts),
        metadatas=[{"category": c} for _, c in DOCUMENTS],
    )
    print(f"Collection: {col.name} | metric: cosine | count: {col.count()}")
    return col


def words(text: str) -> set[str]:
    """Lowercase word set without stopwords (used for the keyword baseline)."""
    return set(re.findall(r"[a-z]+", text.lower())) - STOPWORDS


def keyword_search(query: str, k: int = 3) -> list[tuple[int, str]]:
    """Baseline: rank documents by number of shared words with the query."""
    q = words(query)
    scored = [(len(q & words(t)), t) for t, _ in DOCUMENTS]
    return sorted(scored, key=lambda s: -s[0])[:k]


def semantic_search(col, query: str, k: int = 3, where: dict | None = None) -> list[tuple[float, str, str]]:
    """Return [(similarity, category, text)] for the k nearest documents."""
    res = col.query(query_embeddings=embed([query]), n_results=k, where=where)
    return [
        (1 - dist, meta["category"], doc)
        for doc, dist, meta in zip(
            res["documents"][0], res["distances"][0], res["metadatas"][0]
        )
    ]


def compare(col, query: str) -> None:
    """Part 1 and 2: keyword vs semantic, with a similarity threshold."""
    print(f"\n{'=' * 70}\nQuery: {query!r}")

    print("\n-- Keyword search (shared words) --")
    for score, text in keyword_search(query):
        print(f"   overlap={score}  {text}")

    print("\n-- Semantic search (top 5, cosine) --")
    hits = semantic_search(col, query, k=5)
    for sim, cat, text in hits:
        keep = "KEEP" if sim >= MIN_SIM else "drop"
        print(f"   sim={sim:.3f} [{cat}] ({keep} @ {MIN_SIM}) {text}")


def filters(col, query: str) -> None:
    """Part 3: combined metadata filters."""
    print(f"\n{'=' * 70}\nFilters for: {query!r}")

    print("\n-- category in [ml, devops] --")
    for sim, cat, text in semantic_search(
        col, query, where={"category": {"$in": ["ml", "devops"]}}
    ):
        print(f"   sim={sim:.3f} [{cat}] {text}")

    print("\n-- category = programming AND NOT ml (via $and / $ne) --")
    for sim, cat, text in semantic_search(
        col, query,
        where={"$and": [{"category": {"$eq": "programming"}}, {"category": {"$ne": "ml"}}]},
    ):
        print(f"   sim={sim:.3f} [{cat}] {text}")


def main() -> None:
    col = build_collection()
    queries = [sys.argv[1]] if len(sys.argv) > 1 else QUERIES

    for q in queries:
        compare(col, q)
    filters(col, queries[0])

    print("\nVERIFY: for each query, does semantic search find the right document "
          "even when keyword overlap is 0? Check each result by eye.")


if __name__ == "__main__":
    main()