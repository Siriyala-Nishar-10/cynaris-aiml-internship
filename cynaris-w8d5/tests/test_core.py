"""Offline unit tests: no Ollama, CrewAI, LangGraph or MLflow required."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from research_assistant.config import Settings
from research_assistant.graph import route_after_retrieval
from research_assistant.memory import Chunk, VectorMemory, chunk_text, cosine, ingest_text


class CoreTests(unittest.TestCase):
    def test_cosine_identical_vectors(self):
        self.assertAlmostEqual(cosine([1, 2], [1, 2]), 1.0)

    def test_cosine_zero_vector_is_safe(self):
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)

    def test_chunking_overlaps(self):
        self.assertEqual(len(chunk_text("a" * 1000, 800, 100)), 2)

    def test_chunking_rejects_bad_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", 10, 10)

    def test_search_ranks_by_similarity(self):
        with tempfile.TemporaryDirectory() as d:
            mem = VectorMemory(os.path.join(d, "m.json"))
            mem.add(Chunk("a", "cats", [1, 0]))
            mem.add(Chunk("b", "dogs", [0, 1]))
            self.assertEqual(mem.search([1, 0], 1)[0].source, "a")

    def test_store_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            mem = VectorMemory(path)
            mem.add(Chunk("a", "x", [1.0]))
            mem.save()
            self.assertEqual(len(VectorMemory(path).chunks), 1)

    def test_routing(self):
        self.assertEqual(route_after_retrieval({"contexts": ["x"]}), "research")
        self.assertEqual(route_after_retrieval({"contexts": []}), "fallback")

    def test_settings_read_env(self):
        os.environ["TOP_K"] = "9"
        try:
            self.assertEqual(Settings().top_k, 9)
        finally:
            del os.environ["TOP_K"]

    def test_ingest_text_uses_embedder(self):
        with tempfile.TemporaryDirectory() as d:
            mem = VectorMemory(os.path.join(d, "m.json"))
            n = ingest_text("doc.md", "hello world", mem, Settings(),
                            embed_fn=lambda text, cfg: [1.0, 0.0])
            self.assertEqual(n, 1)
            self.assertEqual(mem.chunks[0].source, "doc.md")


if __name__ == "__main__":
    unittest.main()
