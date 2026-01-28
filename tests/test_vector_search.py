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


class TestVectorIndex(unittest.TestCase):
    """Test VectorIndex class."""

    def test_add_and_get(self):
        """VectorIndex should store and retrieve embeddings."""
        from rag_system.search.vector_search import VectorIndex

        index = VectorIndex()
        embedding = [1.0, 2.0, 3.0]
        index.add(1, embedding)

        self.assertEqual(index.get(1), embedding)
        self.assertEqual(index.count, 1)

    def test_remove(self):
        """VectorIndex should remove embeddings."""
        from rag_system.search.vector_search import VectorIndex

        index = VectorIndex()
        index.add(1, [1.0, 2.0])
        index.add(2, [3.0, 4.0])

        index.remove(1)

        self.assertIsNone(index.get(1))
        self.assertEqual(index.count, 1)

    def test_clear(self):
        """VectorIndex should clear all embeddings."""
        from rag_system.search.vector_search import VectorIndex

        index = VectorIndex()
        index.add(1, [1.0, 2.0])
        index.add(2, [3.0, 4.0])

        index.clear()

        self.assertEqual(index.count, 0)

    def test_items_iterator(self):
        """VectorIndex.items should iterate over all embeddings."""
        from rag_system.search.vector_search import VectorIndex

        index = VectorIndex()
        index.add(1, [1.0, 2.0])
        index.add(2, [3.0, 4.0])

        items = list(index.items())

        self.assertEqual(len(items), 2)
        chunk_ids = [item[0] for item in items]
        self.assertIn(1, chunk_ids)
        self.assertIn(2, chunk_ids)

    def test_save_and_load(self):
        """VectorIndex should save and load from file."""
        from rag_system.search.vector_search import VectorIndex

        index = VectorIndex()
        index.add(1, [1.0, 2.0, 3.0])
        index.add(2, [4.0, 5.0, 6.0])
        index.content_hash = "test_hash"

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            index.save(filepath)
            loaded = VectorIndex.load(filepath)

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.count, 2)
            self.assertEqual(loaded.content_hash, "test_hash")
            self.assertEqual(loaded.get(1), [1.0, 2.0, 3.0])
            self.assertEqual(loaded.get(2), [4.0, 5.0, 6.0])
        finally:
            os.unlink(filepath)

    def test_load_nonexistent_file(self):
        """VectorIndex.load should return None for nonexistent file."""
        from rag_system.search.vector_search import VectorIndex

        result = VectorIndex.load('/nonexistent/path/index.json')

        self.assertIsNone(result)

    def test_load_invalid_version(self):
        """VectorIndex.load should return None for wrong version."""
        from rag_system.search.vector_search import VectorIndex

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'version': 999, 'embeddings': {}}, f)
            filepath = f.name

        try:
            result = VectorIndex.load(filepath)
            self.assertIsNone(result)
        finally:
            os.unlink(filepath)


class TestVectorSearchPersistence(unittest.TestCase):
    """Test vector search index persistence."""

    def setUp(self):
        """Create temporary database with test data."""
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
        update_chunk_embedding(conn, self.chunk_id, [1.0, 0.0, 0.0])
        conn.close()

        # Create temp file for index
        self.temp_fd2, self.index_path = tempfile.mkstemp(suffix='.json')
        os.close(self.temp_fd2)
        os.unlink(self.index_path)  # Remove it so VectorSearch creates it

    def tearDown(self):
        """Remove temporary files."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)
        if os.path.exists(self.index_path):
            os.unlink(self.index_path)

    def test_search_creates_index_file(self):
        """VectorSearch should create index file on first search."""
        from rag_system.search.vector_search import VectorSearch

        searcher = VectorSearch(self.temp_path, index_path=self.index_path)
        searcher.search([1.0, 0.0, 0.0])

        self.assertTrue(os.path.exists(self.index_path))

    def test_search_loads_from_index_file(self):
        """VectorSearch should load from index file on subsequent runs."""
        from rag_system.search.vector_search import VectorSearch, VectorIndex

        # First run - builds and saves index
        searcher1 = VectorSearch(self.temp_path, index_path=self.index_path)
        searcher1.search([1.0, 0.0, 0.0])

        # Second run - should load from file
        searcher2 = VectorSearch(self.temp_path, index_path=self.index_path)
        results = searcher2.search([1.0, 0.0, 0.0])

        self.assertGreater(len(results), 0)

    def test_search_invalidates_stale_index(self):
        """VectorSearch should rebuild index when database changes."""
        from rag_system.search.vector_search import VectorSearch
        from rag_system.database import get_connection, insert_chunk, update_chunk_embedding

        # First run - builds index
        searcher1 = VectorSearch(self.temp_path, index_path=self.index_path)
        searcher1.search([1.0, 0.0, 0.0])

        # Add new chunk to database
        conn = get_connection(self.temp_path)
        chunk_id = insert_chunk(conn, page_id=1, chunk_type='small',
                               chunk_index=1, content='new',
                               heading_path='')
        update_chunk_embedding(conn, chunk_id, [0.0, 1.0, 0.0])
        conn.close()

        # Second run with new searcher - should detect changes
        searcher2 = VectorSearch(self.temp_path, index_path=self.index_path)
        results = searcher2.search([0.0, 1.0, 0.0])

        # Should find the new chunk
        chunk_ids = [r[0] for r in results]
        self.assertIn(chunk_id, chunk_ids)


class TestVectorSearchMaxEmbeddings(unittest.TestCase):
    """Test vector search with max_embeddings limit."""

    def setUp(self):
        """Create temporary database with multiple embeddings."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk, update_chunk_embedding
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='', content_hash='abc')

        # Create 10 chunks with embeddings
        self.chunk_ids = []
        for i in range(10):
            chunk_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                   chunk_index=i, content=f'chunk {i}',
                                   heading_path='')
            embedding = [0.0] * 10
            embedding[i] = 1.0
            update_chunk_embedding(conn, chunk_id, embedding)
            self.chunk_ids.append(chunk_id)

        conn.close()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_max_embeddings_limits_loaded(self):
        """VectorSearch should respect max_embeddings limit."""
        from rag_system.search.vector_search import VectorSearch

        # Limit to 5 embeddings
        searcher = VectorSearch(self.temp_path, max_embeddings=5)
        index = searcher._load_index()

        self.assertEqual(index.count, 5)

    def test_search_works_with_limited_embeddings(self):
        """VectorSearch should work correctly with limited embeddings."""
        from rag_system.search.vector_search import VectorSearch

        searcher = VectorSearch(self.temp_path, max_embeddings=5)
        query = [1.0] + [0.0] * 9  # Should match first chunk
        results = searcher.search(query)

        self.assertGreater(len(results), 0)


class TestVectorSearchCacheInvalidation(unittest.TestCase):
    """Test vector search cache invalidation."""

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

    def test_clear_cache(self):
        """clear_cache should clear the embeddings cache."""
        from rag_system.search.vector_search import VectorSearch

        searcher = VectorSearch(self.temp_path)
        searcher.search([1.0, 0.0])  # Load cache

        self.assertIsNotNone(searcher._index)

        searcher.clear_cache()

        self.assertIsNone(searcher._index)

    def test_invalidate_if_changed_detects_changes(self):
        """invalidate_if_changed should detect database changes."""
        from rag_system.search.vector_search import VectorSearch
        from rag_system.database import get_connection, insert_chunk, update_chunk_embedding

        searcher = VectorSearch(self.temp_path)
        searcher.search([1.0, 0.0])  # Load cache

        # Add new chunk
        conn = get_connection(self.temp_path)
        chunk_id = insert_chunk(conn, page_id=1, chunk_type='small',
                               chunk_index=1, content='new',
                               heading_path='')
        update_chunk_embedding(conn, chunk_id, [0.0, 1.0])
        conn.close()

        invalidated = searcher.invalidate_if_changed()

        self.assertTrue(invalidated)
        self.assertIsNone(searcher._index)

    def test_invalidate_if_changed_no_change(self):
        """invalidate_if_changed should not invalidate if no changes."""
        from rag_system.search.vector_search import VectorSearch

        searcher = VectorSearch(self.temp_path)
        searcher.search([1.0, 0.0])  # Load cache

        invalidated = searcher.invalidate_if_changed()

        self.assertFalse(invalidated)
        self.assertIsNotNone(searcher._index)


class TestVectorSearchMemoryEstimate(unittest.TestCase):
    """Test memory usage estimation."""

    def setUp(self):
        """Create temporary database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk, update_chunk_embedding
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='', content_hash='abc')

        # Create chunks with 768-dim embeddings (typical size)
        for i in range(5):
            chunk_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                   chunk_index=i, content=f'chunk {i}',
                                   heading_path='')
            update_chunk_embedding(conn, chunk_id, [0.1] * 768)

        conn.close()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_get_memory_usage_estimate(self):
        """get_memory_usage_estimate should return memory info."""
        from rag_system.search.vector_search import VectorSearch

        searcher = VectorSearch(self.temp_path)
        info = searcher.get_memory_usage_estimate()

        self.assertEqual(info['embedding_count'], 5)
        self.assertEqual(info['dimension'], 768)
        self.assertGreater(info['estimated_mb'], 0)
        self.assertFalse(info['is_limited'])

    def test_get_memory_usage_estimate_empty(self):
        """get_memory_usage_estimate should handle empty index."""
        from rag_system.search.vector_search import VectorSearch

        # Create empty database
        temp_fd, empty_path = tempfile.mkstemp(suffix='.db')
        os.close(temp_fd)
        from rag_system.database import init_db
        init_db(empty_path)

        try:
            searcher = VectorSearch(empty_path)
            info = searcher.get_memory_usage_estimate()

            self.assertEqual(info['embedding_count'], 0)
            self.assertEqual(info['estimated_mb'], 0)
        finally:
            os.unlink(empty_path)


if __name__ == '__main__':
    unittest.main()
