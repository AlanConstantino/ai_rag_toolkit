"""Performance benchmarks for the RAG system.

Run with: python -m unittest tests.test_benchmarks -v
"""

import os
import tempfile
import time
import unittest

from rag_system.database import (
    init_db, get_connection, insert_page, insert_chunk
)
from rag_system.search.bm25_search import BM25Index, BM25Search
from rag_system.search.vector_search import VectorSearch, cosine_similarity


class TestBM25IndexBenchmarks(unittest.TestCase):
    """Benchmarks for BM25 index building."""

    def setUp(self):
        """Create temporary database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_index_build_100_chunks(self):
        """Benchmark: Build index with 100 chunks."""
        conn = get_connection(self.temp_path)
        page_id = insert_page(
            conn, url='http://test.com', title='Test',
            raw_html='', parsed_text='', content_hash='abc'
        )
        for i in range(100):
            insert_chunk(
                conn, page_id=page_id, chunk_type='small',
                chunk_index=i,
                content=f'chunk {i} with some python programming content about databases',
                heading_path=f'Section {i}'
            )
        conn.close()

        start = time.time()
        index = BM25Index(self.temp_path)
        index.build()
        elapsed = time.time() - start

        # Should complete in under 1 second
        self.assertLess(elapsed, 1.0, f"Index build took {elapsed:.3f}s")

    def test_index_build_1000_chunks(self):
        """Benchmark: Build index with 1000 chunks."""
        conn = get_connection(self.temp_path)
        page_id = insert_page(
            conn, url='http://test.com', title='Test',
            raw_html='', parsed_text='', content_hash='abc'
        )
        for i in range(1000):
            insert_chunk(
                conn, page_id=page_id, chunk_type='small',
                chunk_index=i,
                content=f'chunk {i} with some python programming content about databases',
                heading_path=f'Section {i}'
            )
        conn.close()

        start = time.time()
        index = BM25Index(self.temp_path)
        index.build()
        elapsed = time.time() - start

        # Should complete in under 5 seconds
        self.assertLess(elapsed, 5.0, f"Index build took {elapsed:.3f}s")


class TestBM25SearchBenchmarks(unittest.TestCase):
    """Benchmarks for BM25 search operations."""

    @classmethod
    def setUpClass(cls):
        """Create shared test database with many chunks."""
        cls.temp_fd, cls.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(cls.temp_fd)
        init_db(cls.temp_path)

        conn = get_connection(cls.temp_path)
        page_id = insert_page(
            conn, url='http://test.com', title='Test',
            raw_html='', parsed_text='', content_hash='abc'
        )

        # Create 500 varied chunks for realistic testing
        topics = ['python', 'database', 'networking', 'security', 'performance']
        for i in range(500):
            topic = topics[i % len(topics)]
            insert_chunk(
                conn, page_id=page_id, chunk_type='small',
                chunk_index=i,
                content=f'documentation about {topic} feature {i} with implementation details',
                heading_path=f'{topic.title()} > Feature {i}'
            )
        conn.close()

        # Build index
        index = BM25Index(cls.temp_path)
        index.build()

    @classmethod
    def tearDownClass(cls):
        """Remove shared test database."""
        if os.path.exists(cls.temp_path):
            os.unlink(cls.temp_path)

    def test_single_query_latency(self):
        """Benchmark: Single query should be fast."""
        searcher = BM25Search(self.temp_path)

        start = time.time()
        results = searcher.search('python documentation', top_k=10)
        elapsed = time.time() - start

        # Single query should be under 100ms
        self.assertLess(elapsed, 0.1, f"Query took {elapsed*1000:.1f}ms")
        self.assertGreater(len(results), 0)

    def test_100_queries_throughput(self):
        """Benchmark: 100 queries should complete quickly."""
        searcher = BM25Search(self.temp_path)
        queries = [
            'python documentation',
            'database performance',
            'networking security',
            'implementation details',
            'feature configuration',
        ]

        start = time.time()
        for i in range(100):
            query = queries[i % len(queries)]
            results = searcher.search(query, top_k=10)
        elapsed = time.time() - start

        # 100 queries should complete in under 2 seconds
        self.assertLess(elapsed, 2.0, f"100 queries took {elapsed:.2f}s")
        qps = 100 / elapsed
        # Should achieve at least 50 QPS
        self.assertGreater(qps, 50, f"Only achieved {qps:.1f} QPS")


class TestCosineSimilarityBenchmarks(unittest.TestCase):
    """Benchmarks for vector similarity calculations."""

    def test_similarity_1000_vectors(self):
        """Benchmark: Calculate similarity with 1000 vectors."""
        import random

        # Create query vector and 1000 document vectors (768 dimensions)
        query_vec = [random.random() for _ in range(768)]
        doc_vecs = [[random.random() for _ in range(768)] for _ in range(1000)]

        start = time.time()
        results = []
        for i, doc_vec in enumerate(doc_vecs):
            sim = cosine_similarity(query_vec, doc_vec)
            results.append((i, sim))
        elapsed = time.time() - start

        # Should complete in under 1 second
        self.assertLess(elapsed, 1.0, f"1000 similarity calculations took {elapsed:.3f}s")
        self.assertEqual(len(results), 1000)


class TestConcurrentQueryBenchmarks(unittest.TestCase):
    """Benchmarks for concurrent query handling."""

    @classmethod
    def setUpClass(cls):
        """Create shared test database."""
        cls.temp_fd, cls.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(cls.temp_fd)
        init_db(cls.temp_path)

        conn = get_connection(cls.temp_path)
        page_id = insert_page(
            conn, url='http://test.com', title='Test',
            raw_html='', parsed_text='', content_hash='abc'
        )
        for i in range(200):
            insert_chunk(
                conn, page_id=page_id, chunk_type='small',
                chunk_index=i,
                content=f'documentation content chunk {i} with various topics',
                heading_path=f'Section {i}'
            )
        conn.close()

        index = BM25Index(cls.temp_path)
        index.build()

    @classmethod
    def tearDownClass(cls):
        """Remove shared test database."""
        if os.path.exists(cls.temp_path):
            os.unlink(cls.temp_path)

    def test_concurrent_queries(self):
        """Benchmark: Multiple threads querying concurrently."""
        import threading

        results = []
        errors = []

        def run_queries(thread_id, num_queries):
            try:
                searcher = BM25Search(self.temp_path)
                for i in range(num_queries):
                    r = searcher.search('documentation content', top_k=10)
                    results.append((thread_id, len(r)))
            except Exception as e:
                errors.append((thread_id, e))

        # Run 4 threads, each doing 25 queries
        threads = []
        start = time.time()

        for i in range(4):
            t = threading.Thread(target=run_queries, args=(i, 25))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        elapsed = time.time() - start

        # Should complete all 100 queries
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 100)

        # Should complete in under 3 seconds
        self.assertLess(elapsed, 3.0, f"Concurrent queries took {elapsed:.2f}s")


class TestMemoryBenchmarks(unittest.TestCase):
    """Basic memory usage tests."""

    def setUp(self):
        """Create temporary database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_vector_search_memory_estimate(self):
        """Memory estimate should be accurate."""
        import json

        conn = get_connection(self.temp_path)
        page_id = insert_page(
            conn, url='http://test.com', title='Test',
            raw_html='', parsed_text='', content_hash='abc'
        )

        # Insert 100 chunks with embeddings
        embedding = [0.1] * 768  # 768-dim vector
        embedding_json = json.dumps(embedding)

        for i in range(100):
            conn.execute("""
                INSERT INTO chunks (page_id, chunk_type, chunk_index, content,
                                   heading_path, embedding_json)
                VALUES (?, 'small', ?, ?, '', ?)
            """, (page_id, i, f'content {i}', embedding_json))
        conn.commit()
        conn.close()

        search = VectorSearch(self.temp_path)
        usage = search.get_memory_usage_estimate()

        self.assertEqual(usage['embedding_count'], 100)
        self.assertEqual(usage['dimension'], 768)
        # Each 768-dim embedding should use roughly 6KB
        self.assertGreater(usage['estimated_mb'], 0.5)
        self.assertLess(usage['estimated_mb'], 1.5)


if __name__ == '__main__':
    unittest.main()
