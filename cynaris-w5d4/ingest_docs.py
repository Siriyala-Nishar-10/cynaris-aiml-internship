"""
W5D4: Vector store setup with ChromaDB (cosine collection, 20 docs, search)
----------------------------------------------------------------------------
Foundation for the rest of Week 5 Day 4. Embeds 20 short documents with
Ollama and stores them in ChromaDB, then shows the two basic retrieval steps:

  1. Collection + embeddings: a persistent cosine collection holding 20
     documents, each with a `category` metadata field.
  2. Similarity search: embed the query with the SAME model and return the
     nearest documents with their cosine similarity scores.
  3. Metadata filter: restrict the search with `where`, so only documents in
     one category can be returned.

Why cosine? Chroma defaults to L2 distance. The metric is fixed when the
collection is created (`hnsw:space`), so it must be set up front.
Chroma returns cosine DISTANCE, so similarity = 1 - distance.

Run:  python ingest_docs.py
Needs: pip install chromadb ollama   |   ollama pull nomic-embed-text
Also imported by semantic_search.py (for DOCUMENTS, DB_PATH and embed).
"""

import chromadb
import ollama

DB_PATH = "./chroma_db"
COLLECTION = "docs"
EMBED_MODEL = "nomic-embed-text"

# (text, category) - 20 documents across 4 categories
DOCUMENTS = [
    # ml
    ("Neural networks learn by adjusting weights through backpropagation.", "ml"),
    ("Gradient descent minimizes a loss function by following its slope.", "ml"),
    ("Transformers use self-attention to model relationships in a sequence.", "ml"),
    ("Overfitting happens when a model memorizes training data instead of generalizing.", "ml"),
    ("Embeddings map text to dense vectors where similar meanings sit close together.", "ml"),
    # programming
    ("Python is a popular language for data science and scripting.", "programming"),
    ("Pandas provides DataFrame structures for working with tabular data.", "programming"),
    ("A list comprehension builds a new list from an iterable in one line.", "programming"),
    ("Virtual environments isolate project dependencies from the system Python.", "programming"),
    ("Unit tests check that small pieces of code behave as expected.", "programming"),
    # devops
    ("Docker packages applications and dependencies into portable containers.", "devops"),
    ("Kubernetes orchestrates containers across a cluster of machines.", "devops"),
    ("Git tracks changes to source code and supports branching and merging.", "devops"),
    ("A CI pipeline automatically builds and tests code on every push.", "devops"),
    ("MLflow tracks experiments, parameters, metrics and model artifacts.", "devops"),
    # geography
    ("Paris is the capital of France and sits on the river Seine.", "geography"),
    ("Mount Everest is the highest mountain above sea level.", "geography"),
    ("The Amazon rainforest spans several countries in South America.", "geography"),
    ("Bengaluru is the capital of the Indian state of Karnataka.", "geography"),
    ("The Sahara is the largest hot desert in the world.", "geography"),
]

QUERY = "How do models learn from data?"


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts; returns one vector per text, in order."""
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def show(label: str, res: dict) -> None:
    """Print ranked results with cosine similarity and category."""
    print(f"\n=== {label} ===")
    for rank, (doc, dist, meta) in enumerate(
        zip(res["documents"][0], res["distances"][0], res["metadatas"][0]), start=1
    ):
        print(f"{rank}. sim={1 - dist:.3f} [{meta['category']}] {doc}")


def main() -> None:
    # Task 1: collection + 20 documents with embeddings
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    texts = [t for t, _ in DOCUMENTS]
    col.upsert(  # upsert = safe to re-run, no duplicate-ID errors
        ids=[f"doc{i}" for i in range(len(DOCUMENTS))],
        documents=texts,
        embeddings=embed(texts),
        metadatas=[{"category": c} for _, c in DOCUMENTS],
    )
    print(f"Collection: {col.name} | metric: cosine | count: {col.count()}")

    # Task 2a: similarity search
    print(f"\nQuery: {QUERY!r}")
    q_vec = embed([QUERY])
    res = col.query(query_embeddings=q_vec, n_results=3)
    show("Similarity search (top 3)", res)
    print("VERIFY: do the top results describe how models learn? (check by eye)")

    # Task 2b: metadata filtering
    res = col.query(
        query_embeddings=q_vec, n_results=3, where={"category": "geography"}
    )
    show("Filtered: category = geography", res)
    print("VERIFY: every result above must be category=geography; scores should be low")


if __name__ == "__main__":
    main()