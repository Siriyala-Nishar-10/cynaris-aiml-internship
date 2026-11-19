"""W6D4 Tasks 1 & 2: ChromaDB setup, 20 documents, cosine search, metadata filter.

Prereqs:
    pip install -r requirements.txt
    ollama pull nomic-embed-text
"""
import chromadb
from langchain_ollama import OllamaEmbeddings

COLLECTION = "docs"

# 20 short documents across 4 topics (5 each) so filtering is easy to verify.
DOCS = {
    "ml": [
        "Gradient descent updates model weights by following the negative gradient of the loss.",
        "Overfitting happens when a model memorises training data and fails to generalise.",
        "A neural network is built from layers of neurons with non-linear activation functions.",
        "Cross-validation splits data into folds to estimate how well a model will perform.",
        "Transformers use self-attention to model relationships between tokens in a sequence.",
    ],
    "python": [
        "A Python list comprehension builds a new list from an iterable in a single expression.",
        "Decorators wrap a function to add behaviour without changing its source code.",
        "Virtual environments isolate project dependencies so packages do not conflict.",
        "Generators use the yield keyword to produce values lazily, saving memory.",
        "The with statement manages resources like files and ensures they are closed properly.",
    ],
    "cooking": [
        "Sear the steak on a very hot pan to develop a brown crust through the Maillard reaction.",
        "Let bread dough rise in a warm place until it doubles in size.",
        "Salt pasta water generously so the noodles absorb flavour as they cook.",
        "Rest cooked meat for a few minutes so the juices redistribute through the cut.",
        "Toast whole spices in a dry pan to release their aroma before grinding.",
    ],
    "space": [
        "A black hole is a region of spacetime where gravity is so strong that light cannot escape.",
        "The Moon orbits the Earth roughly every 27 days and causes the ocean tides.",
        "Mars has a thin atmosphere made mostly of carbon dioxide and a reddish, iron-rich surface.",
        "A light-year is the distance light travels in one year, about 9.46 trillion kilometres.",
        "The International Space Station orbits Earth about every 90 minutes.",
    ],
}

emb = OllamaEmbeddings(model="nomic-embed-text")
client = chromadb.PersistentClient(path="./chroma_db")


def make_collection(name: str):
    """Create a fresh cosine-space collection (works across Chroma versions)."""
    try:
        client.delete_collection(name)
    except Exception:
        pass
    # Chroma defaults to L2 distance, so cosine must be requested explicitly.
    try:
        return client.create_collection(
            name, configuration={"hnsw": {"space": "cosine"}}
        )
    except Exception:
        # Older Chroma versions use metadata instead of configuration.
        return client.create_collection(name, metadata={"hnsw:space": "cosine"})


col = make_collection(COLLECTION)


def build_collection():
    texts, metas = [], []
    for topic, items in DOCS.items():
        for t in items:
            texts.append(t)
            metas.append({"topic": topic})
    col.add(
        ids=[f"doc{i}" for i in range(len(texts))],
        documents=texts,
        embeddings=emb.embed_documents(texts),
        metadatas=metas,
    )
    print(f"Added {col.count()} documents.\n")


def show(title, res):
    print(f"--- {title} ---")
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        # With cosine space, Chroma distance = 1 - cosine similarity.
        print(f"  sim={1 - dist:.3f} [{meta['topic']}] {doc}")
    print()


def search(query, n=3, topic=None):
    kwargs = {"where": {"topic": topic}} if topic else {}
    return col.query(
        query_embeddings=[emb.embed_query(query)], n_results=n, **kwargs
    )


if __name__ == "__main__":
    build_collection()

    # Task 2a: plain cosine similarity search.
    show("Query: 'How do models learn from data?'",
         search("How do models learn from data?"))
    show("Query: 'What is the best way to cook meat?'",
         search("What is the best way to cook meat?"))

    # Task 2b: metadata filtering. Same query, different topic filters.
    q = "How does something work step by step?"
    show("Filtered to topic=python", search(q, topic="python"))
    show("Filtered to topic=cooking", search(q, topic="cooking"))

    # Manual verification: eyeball that (1) top hits match the question's topic
    # and (2) filtered results ONLY contain the requested topic.
    res = search(q, topic="space")
    assert all(m["topic"] == "space" for m in res["metadatas"][0]), "Filter leaked!"
    print("Filter check passed: only 'space' docs returned.")
