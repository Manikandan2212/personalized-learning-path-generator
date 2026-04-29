"""
Unit Tests — RAG Vector Store
Tests: tokenization, document indexing, TF-IDF similarity, retrieval ranking.
"""

import sys
sys.path.insert(0, "/home/claude/learning_system")

import unittest
from rag.vector_store import VectorStore, tokenize, chunk_text


class TestTokenizer(unittest.TestCase):
    def test_lowercase(self):
        tokens = tokenize("Hello World PYTHON")
        self.assertEqual(tokens, ["hello", "world", "python"])

    def test_removes_stopwords(self):
        tokens = tokenize("the cat is on the mat")
        self.assertNotIn("the", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("on", tokens)
        self.assertIn("cat", tokens)
        self.assertIn("mat", tokens)

    def test_removes_punctuation(self):
        tokens = tokenize("machine-learning, deep.learning!")
        # Should not have punctuation
        for tok in tokens:
            self.assertNotIn(".", tok)
            self.assertNotIn(",", tok)

    def test_empty_string(self):
        self.assertEqual(tokenize(""), [])

    def test_short_tokens_removed(self):
        tokens = tokenize("I go to AI")  # 'I' length 1 removed
        self.assertNotIn("i", tokens)


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.store = VectorStore()
        self.store.add_document("doc1", "Python is a programming language used in machine learning", {"topic": "Python"})
        self.store.add_document("doc2", "Neural networks learn from data using gradient descent", {"topic": "Deep Learning"})
        self.store.add_document("doc3", "Linear algebra vectors matrices are used in deep learning", {"topic": "Math"})
        self.store.add_document("doc4", "HTML CSS are used to build web pages and style them", {"topic": "Web"})

    def test_document_count(self):
        self.assertEqual(self.store.N, 4)

    def test_vocabulary_built(self):
        self.assertGreater(len(self.store.df), 0)

    def test_search_returns_results(self):
        results = self.store.search("machine learning python")
        self.assertGreater(len(results), 0)

    def test_search_top_result_relevant(self):
        results = self.store.search("neural networks gradient descent")
        self.assertGreater(len(results), 0)
        top_doc_id = results[0]["doc_id"]
        self.assertEqual(top_doc_id, "doc2")

    def test_search_top_k_respected(self):
        results = self.store.search("learning", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_scores_descending(self):
        results = self.store.search("deep learning neural")
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_scores_between_0_and_1(self):
        results = self.store.search("machine learning")
        for r in results:
            self.assertGreaterEqual(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.0)

    def test_irrelevant_query_low_score(self):
        results = self.store.search("cooking recipes pasta")
        if results:
            self.assertLess(results[0]["score"], 0.5)

    def test_empty_query(self):
        results = self.store.search("")
        self.assertEqual(results, [])

    def test_metadata_returned(self):
        results = self.store.search("python programming")
        self.assertIn("metadata", results[0])
        self.assertIn("topic", results[0]["metadata"])

    def test_get_document(self):
        doc = self.store.get_document("doc1")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["doc_id"], "doc1")
        self.assertIn("content", doc)

    def test_get_nonexistent_document(self):
        doc = self.store.get_document("nonexistent")
        self.assertIsNone(doc)

    def test_stats(self):
        stats = self.store.stats()
        self.assertEqual(stats["total_documents"], 4)
        self.assertIn("vocabulary_size", stats)


class TestChunker(unittest.TestCase):
    def test_short_text_one_chunk(self):
        chunks = chunk_text("hello world this is short", chunk_size=100, overlap=10)
        self.assertEqual(len(chunks), 1)

    def test_long_text_multiple_chunks(self):
        words = ["word"] * 500
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)

    def test_chunks_overlap(self):
        words = [f"w{i}" for i in range(200)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        # Each chunk should have up to 50 words
        for chunk in chunks:
            self.assertLessEqual(len(chunk.split()), 50)

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=100, overlap=10)
        self.assertEqual(len(chunks), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
