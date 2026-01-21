"""Tests for the reranker module."""

import unittest


class TestReranker(unittest.TestCase):
    """Test reranking functionality."""

    def test_rerank_by_score(self):
        """rerank should sort by combined score."""
        from rag_system.search.reranker import rerank

        results = [(1, 0.5), (2, 0.8), (3, 0.6)]
        reranked = rerank(results)

        self.assertEqual(reranked[0][0], 2)  # Highest score first

    def test_rerank_with_boost(self):
        """rerank should apply boost factors."""
        from rag_system.search.reranker import rerank

        results = [(1, 0.8), (2, 0.7)]
        boost_factors = {2: 1.5}  # Boost chunk 2

        reranked = rerank(results, boost_factors=boost_factors)

        # Chunk 2 should now be higher: 0.7 * 1.5 = 1.05 > 0.8
        self.assertEqual(reranked[0][0], 2)

    def test_rerank_respects_top_k(self):
        """rerank should limit results."""
        from rag_system.search.reranker import rerank

        results = [(i, 0.9 - i * 0.1) for i in range(10)]
        reranked = rerank(results, top_k=3)

        self.assertEqual(len(reranked), 3)


class TestContextualReranker(unittest.TestCase):
    """Test contextual reranking with heading paths."""

    def test_rerank_boosts_heading_match(self):
        """Reranker should boost results with matching heading path."""
        from rag_system.search.reranker import Reranker
        import tempfile
        import os
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk

        temp_fd, temp_path = tempfile.mkstemp(suffix='.db')
        os.close(temp_fd)
        init_db(temp_path)

        conn = get_connection(temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='', content_hash='a')
        chunk1_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                chunk_index=0, content='general content',
                                heading_path='Introduction')
        chunk2_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                chunk_index=1, content='config content',
                                heading_path='Configuration > Settings')
        conn.close()

        try:
            reranker = Reranker(temp_path)
            results = [(chunk1_id, 0.8), (chunk2_id, 0.7)]

            # Query about configuration should boost chunk2
            reranked = reranker.rerank(
                results,
                query='how to configure settings',
                top_k=10
            )

            # Chunk 2 should be boosted (has config in heading)
            chunk_ids = [r[0] for r in reranked]
            # First result should be chunk2 due to heading match
            self.assertEqual(chunk_ids[0], chunk2_id)
        finally:
            os.unlink(temp_path)


class TestRerankerWithoutContext(unittest.TestCase):
    """Test reranker without database context."""

    def test_reranker_passthrough(self):
        """Reranker without DB should pass through results."""
        from rag_system.search.reranker import Reranker

        reranker = Reranker(db_path=None)
        results = [(1, 0.9), (2, 0.8)]

        reranked = reranker.rerank(results, query='test', top_k=10)

        # Should maintain order by score
        self.assertEqual([r[0] for r in reranked], [1, 2])


if __name__ == '__main__':
    unittest.main()
