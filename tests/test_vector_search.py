"""Tests for the vector search module."""

import unittest
import tempfile
import os
import json


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity calculation."""

    def test_cosine_similarity_identical(self):
        """cosine_similarity should return 1.0 for identical vectors."""
        from rag_system.search.vector_search import cosine_similarity

        vec = [1.0, 2.0, 3.0]
        result = cosine_similarity(vec, vec)

        self.assertAlmostEqual(result, 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):
        """cosine_similarity should return 0.0 for orthogonal vectors."""
        from rag_system.search.vector_search import cosine_similarity

        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        result = cosine_similarity(vec_a, vec_b)

        self.assertAlmostEqual(result, 0.0, places=5)

    def test_cosine_similarity_opposite(self):
        """cosine_similarity should return -1.0 for opposite vectors."""
        from rag_system.search.vector_search import cosine_similarity

        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]
        result = cosine_similarity(vec_a, vec_b)

        self.assertAlmostEqual(result, -1.0, places=5)

    def test_cosine_similarity_zero_vector(self):
        """cosine_similarity should return 0.0 for zero vectors."""
        from rag_system.search.vector_search import cosine_similarity

        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        result = cosine_similarity(vec_a, vec_b)

        self.assertAlmostEqual(result, 0.0, places=5)


class TestVectorSearch(unittest.TestCase):
    """Test vector search functionality."""

    def setUp(self):
        """Create temporary database with test data."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk, update_chunk_embedding
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='', content_hash='abc')

        # Insert chunks with embeddings
        # Chunk 1: about python
        self.chunk1_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=0, content='python programming',
                                      heading_path='')
        update_chunk_embedding(conn, self.chunk1_id, [1.0, 0.0, 0.0])

        # Chunk 2: similar to python
        self.chunk2_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=1, content='coding in python',
                                      heading_path='')
        update_chunk_embedding(conn, self.chunk2_id, [0.9, 0.1, 0.0])

        # Chunk 3: different topic
        self.chunk3_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=2, content='database design',
                                      heading_path='')
        update_chunk_embedding(conn, self.chunk3_id, [0.0, 0.0, 1.0])

        conn.close()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_search_returns_results(self):
        """search should return matching chunks."""
        from rag_system.search.vector_search import VectorSearch

        searcher = VectorSearch(self.temp_path)
        # Query vector similar to python chunks
        query_embedding = [0.95, 0.05, 0.0]
        results = searcher.search(query_embedding, top_k=10)

        self.assertGreater(len(results), 0)

    def test_search_ranks_by_similarity(self):
        """search should rank results by cosine similarity."""
        from rag_system.search.vector_search import VectorSearch

        searcher = VectorSearch(self.temp_path)
        query_embedding = [1.0, 0.0, 0.0]  # Most similar to chunk1
        results = searcher.search(query_embedding, top_k=10)

        # First result should be chunk1
        self.assertEqual(results[0][0], self.chunk1_id)

        # Scores should be sorted descending
        scores = [r[1] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_respects_top_k(self):
        """search should limit results to top_k."""
        from rag_system.search.vector_search import VectorSearch

        searcher = VectorSearch(self.temp_path)
        query_embedding = [0.5, 0.5, 0.5]
        results = searcher.search(query_embedding, top_k=2)

        self.assertLessEqual(len(results), 2)

    def test_search_filters_by_threshold(self):
        """search should filter results below similarity threshold."""
        from rag_system.search.vector_search import VectorSearch

        searcher = VectorSearch(self.temp_path)
        # Query similar to chunk3, dissimilar to chunk1/2
        query_embedding = [0.0, 0.0, 1.0]
        results = searcher.search(query_embedding, top_k=10, threshold=0.5)

        # Should only return chunk3 (high similarity)
        chunk_ids = [r[0] for r in results]
        self.assertIn(self.chunk3_id, chunk_ids)
        # chunk1 has 0 similarity, should be filtered out
        self.assertNotIn(self.chunk1_id, chunk_ids)


class TestVectorSearchWithClient(unittest.TestCase):
    """Test vector search with API client."""

    def setUp(self):
        """Create temporary database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk, update_chunk_embedding
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='', content_hash='abc')
        self.chunk_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                    chunk_index=0, content='test',
                                    heading_path='')
        update_chunk_embedding(conn, self.chunk_id, [1.0, 0.0])
        conn.close()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_search_with_query_text(self):
        """search_text should use vector client to embed query."""
        from rag_system.search.vector_search import VectorSearch
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.get_embedding.return_value = [0.9, 0.1]

        searcher = VectorSearch(self.temp_path, vector_client=mock_client)
        results = searcher.search_text('test query', top_k=10)

        mock_client.get_embedding.assert_called_once_with('test query')
        self.assertGreater(len(results), 0)


if __name__ == '__main__':
    unittest.main()
