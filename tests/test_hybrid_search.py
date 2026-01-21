"""Tests for the hybrid search module."""

import unittest


class TestScoreNormalization(unittest.TestCase):
    """Test score normalization."""

    def test_normalize_scores_basic(self):
        """normalize_scores should scale to 0-1 range."""
        from rag_system.search.hybrid_search import normalize_scores

        results = [(1, 10.0), (2, 5.0), (3, 0.0)]
        normalized = normalize_scores(results)

        # Max should be 1.0, min should be 0.0
        scores = {r[0]: r[1] for r in normalized}
        self.assertAlmostEqual(scores[1], 1.0, places=5)
        self.assertAlmostEqual(scores[3], 0.0, places=5)
        self.assertAlmostEqual(scores[2], 0.5, places=5)

    def test_normalize_scores_empty(self):
        """normalize_scores should handle empty list."""
        from rag_system.search.hybrid_search import normalize_scores

        result = normalize_scores([])
        self.assertEqual(result, [])

    def test_normalize_scores_single(self):
        """normalize_scores should handle single result."""
        from rag_system.search.hybrid_search import normalize_scores

        result = normalize_scores([(1, 5.0)])
        self.assertEqual(result, [(1, 1.0)])

    def test_normalize_scores_identical(self):
        """normalize_scores should handle identical scores."""
        from rag_system.search.hybrid_search import normalize_scores

        results = [(1, 5.0), (2, 5.0)]
        normalized = normalize_scores(results)

        scores = [r[1] for r in normalized]
        self.assertTrue(all(s == 1.0 for s in scores))


class TestHybridMerge(unittest.TestCase):
    """Test hybrid search merging."""

    def test_merge_results_combines_scores(self):
        """merge_results should combine vector and BM25 scores."""
        from rag_system.search.hybrid_search import merge_results

        # Use simple case: both have same chunk with same score
        vector_results = [(1, 1.0)]
        bm25_results = [(1, 1.0)]

        merged = merge_results(
            vector_results, bm25_results,
            vector_weight=0.7, bm25_weight=0.3
        )

        # Chunk 1 should have combined score
        # Both normalize to 1.0 (single result), so combined = 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        chunk_1 = next(r for r in merged if r[0] == 1)
        self.assertAlmostEqual(chunk_1[1], 1.0, places=5)

    def test_merge_results_includes_all_chunks(self):
        """merge_results should include chunks from both sources."""
        from rag_system.search.hybrid_search import merge_results

        vector_results = [(1, 0.9)]
        bm25_results = [(2, 0.8)]

        merged = merge_results(vector_results, bm25_results)

        chunk_ids = [r[0] for r in merged]
        self.assertIn(1, chunk_ids)
        self.assertIn(2, chunk_ids)

    def test_merge_results_sorted_descending(self):
        """merge_results should sort by combined score descending."""
        from rag_system.search.hybrid_search import merge_results

        vector_results = [(1, 0.5), (2, 0.9)]
        bm25_results = [(1, 0.5), (2, 0.9)]

        merged = merge_results(vector_results, bm25_results)

        scores = [r[1] for r in merged]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestHybridSearch(unittest.TestCase):
    """Test HybridSearch class."""

    def test_hybrid_search_combines_methods(self):
        """HybridSearch should combine vector and BM25 search."""
        from rag_system.search.hybrid_search import HybridSearch
        from unittest.mock import MagicMock

        mock_vector_search = MagicMock()
        mock_vector_search.search.return_value = [(1, 0.9), (2, 0.7)]

        mock_bm25_search = MagicMock()
        mock_bm25_search.search.return_value = [(1, 0.8), (3, 0.6)]

        hybrid = HybridSearch(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search
        )

        results = hybrid.search(
            query_embedding=[0.1, 0.2],
            query_text='test query',
            top_k=10
        )

        self.assertGreater(len(results), 0)
        mock_vector_search.search.assert_called_once()
        mock_bm25_search.search.assert_called_once()

    def test_hybrid_search_respects_weights(self):
        """HybridSearch should apply configured weights."""
        from rag_system.search.hybrid_search import HybridSearch
        from unittest.mock import MagicMock

        mock_vector = MagicMock()
        mock_vector.search.return_value = [(1, 1.0)]  # Normalized max

        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [(1, 1.0)]  # Normalized max

        hybrid = HybridSearch(
            vector_search=mock_vector,
            bm25_search=mock_bm25,
            vector_weight=0.6,
            bm25_weight=0.4
        )

        results = hybrid.search([0.1], 'test', top_k=10)

        # Combined score should reflect weights
        # Both normalized to 1.0, so result = 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        self.assertAlmostEqual(results[0][1], 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
