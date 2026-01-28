"""Tests for the BM25 search module."""

import unittest
import tempfile
import os


class TestTokenization(unittest.TestCase):
    """Test tokenization for BM25."""

    def test_tokenize_text(self):
        """tokenize_for_bm25 should split and lowercase text."""
        from rag_system.search.bm25_search import tokenize_for_bm25

        result = tokenize_for_bm25("Hello World Test")

        self.assertEqual(result, ['hello', 'world', 'test'])

    def test_tokenize_removes_stopwords(self):
        """tokenize_for_bm25 should optionally remove stopwords."""
        from rag_system.search.bm25_search import tokenize_for_bm25

        result = tokenize_for_bm25("the quick brown fox", remove_stopwords=True)

        self.assertNotIn('the', result)
        self.assertIn('quick', result)

    def test_tokenize_handles_punctuation(self):
        """tokenize_for_bm25 should handle punctuation."""
        from rag_system.search.bm25_search import tokenize_for_bm25

        result = tokenize_for_bm25("Hello, world! How's it going?")

        self.assertIn('hello', result)
        self.assertIn('world', result)


class TestBM25Scoring(unittest.TestCase):
    """Test BM25 scoring algorithm."""

    def test_bm25_score_basic(self):
        """bm25_score should return a score for matching terms."""
        from rag_system.search.bm25_search import bm25_score

        query_terms = ['hello', 'world']
        doc_term_freqs = {'hello': 2, 'world': 1}
        doc_length = 10
        avg_doc_length = 15.0
        doc_frequencies = {'hello': 5, 'world': 3}
        total_docs = 100

        score = bm25_score(
            query_terms, doc_term_freqs, doc_length,
            avg_doc_length, doc_frequencies, total_docs
        )

        self.assertGreater(score, 0)

    def test_bm25_score_no_match(self):
        """bm25_score should return 0 for no matching terms."""
        from rag_system.search.bm25_search import bm25_score

        query_terms = ['hello']
        doc_term_freqs = {'world': 1}

        score = bm25_score(
            query_terms, doc_term_freqs, 10, 15.0, {'hello': 1}, 100
        )

        self.assertEqual(score, 0.0)

    def test_bm25_score_higher_for_rarer_terms(self):
        """bm25_score should give higher IDF to rarer terms."""
        from rag_system.search.bm25_search import bm25_score

        # Rare term (low doc frequency)
        score_rare = bm25_score(
            ['rare'], {'rare': 1}, 10, 15.0, {'rare': 2}, 1000
        )

        # Common term (high doc frequency)
        score_common = bm25_score(
            ['common'], {'common': 1}, 10, 15.0, {'common': 500}, 1000
        )

        self.assertGreater(score_rare, score_common)


class TestBM25Index(unittest.TestCase):
    """Test BM25 index building."""

    def setUp(self):
        """Create temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_build_bm25_index(self):
        """build_bm25_index should populate term tables."""
        from rag_system.search.bm25_search import BM25Index
        from rag_system.database import get_connection, insert_page, insert_chunk

        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='', content_hash='abc')
        insert_chunk(conn, page_id=page_id, chunk_type='small',
                    chunk_index=0, content='hello world test', heading_path='')
        insert_chunk(conn, page_id=page_id, chunk_type='small',
                    chunk_index=1, content='hello again', heading_path='')
        conn.close()

        index = BM25Index(self.temp_path)
        index.build()

        # Verify corpus stats were created
        conn = get_connection(self.temp_path)
        from rag_system.database import get_corpus_stats
        stats = get_corpus_stats(conn)
        conn.close()

        self.assertIsNotNone(stats)
        self.assertEqual(stats['total_docs'], 2)


class TestBM25Search(unittest.TestCase):
    """Test BM25 search functionality."""

    def setUp(self):
        """Create temporary database with test data."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='', content_hash='abc')
        self.chunk1_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=0, content='python programming tutorial',
                                      heading_path='')
        self.chunk2_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=1, content='java programming guide',
                                      heading_path='')
        self.chunk3_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=2, content='database design patterns',
                                      heading_path='')
        conn.close()

        # Build index
        from rag_system.search.bm25_search import BM25Index
        index = BM25Index(self.temp_path)
        index.build()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_search_returns_results(self):
        """search should return matching chunks."""
        from rag_system.search.bm25_search import BM25Search

        searcher = BM25Search(self.temp_path)
        results = searcher.search('python programming', top_k=10)

        self.assertGreater(len(results), 0)
        # First result should be python chunk
        self.assertEqual(results[0][0], self.chunk1_id)

    def test_search_ranks_by_relevance(self):
        """search should rank results by BM25 score."""
        from rag_system.search.bm25_search import BM25Search

        searcher = BM25Search(self.temp_path)
        results = searcher.search('programming', top_k=10)

        # Results should be sorted by score descending
        scores = [r[1] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_no_results(self):
        """search should return empty list for no matches."""
        from rag_system.search.bm25_search import BM25Search

        searcher = BM25Search(self.temp_path)
        results = searcher.search('nonexistent term xyz', top_k=10)

        self.assertEqual(len(results), 0)

    def test_search_respects_top_k(self):
        """search should limit results to top_k."""
        from rag_system.search.bm25_search import BM25Search

        searcher = BM25Search(self.temp_path)
        results = searcher.search('programming', top_k=1)

        self.assertLessEqual(len(results), 1)


class TestBatchQueries(unittest.TestCase):
    """Test batch query functions for BM25."""

    def setUp(self):
        """Create temporary database with test data."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='', content_hash='abc')
        self.chunk1_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=0, content='python programming',
                                      heading_path='')
        self.chunk2_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=1, content='java programming',
                                      heading_path='')
        self.chunk3_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=2, content='database design',
                                      heading_path='')
        conn.close()

        # Build index
        from rag_system.search.bm25_search import BM25Index
        index = BM25Index(self.temp_path)
        index.build()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_get_doc_terms_batch(self):
        """get_doc_terms_batch should fetch multiple chunks efficiently."""
        from rag_system.database import get_connection, get_doc_terms_batch

        conn = get_connection(self.temp_path)
        try:
            result = get_doc_terms_batch(conn, [self.chunk1_id, self.chunk2_id])

            self.assertIn(self.chunk1_id, result)
            self.assertIn(self.chunk2_id, result)
            self.assertIn('python', result[self.chunk1_id])
            self.assertIn('java', result[self.chunk2_id])
        finally:
            conn.close()

    def test_get_doc_terms_batch_empty(self):
        """get_doc_terms_batch should handle empty input."""
        from rag_system.database import get_connection, get_doc_terms_batch

        conn = get_connection(self.temp_path)
        try:
            result = get_doc_terms_batch(conn, [])
            self.assertEqual(result, {})
        finally:
            conn.close()

    def test_get_term_doc_frequencies_batch(self):
        """get_term_doc_frequencies_batch should fetch multiple terms."""
        from rag_system.database import get_connection, get_term_doc_frequencies_batch

        conn = get_connection(self.temp_path)
        try:
            result = get_term_doc_frequencies_batch(conn, ['python', 'programming', 'nonexistent'])

            self.assertEqual(result['python'], 1)
            self.assertEqual(result['programming'], 2)
            self.assertEqual(result['nonexistent'], 0)
        finally:
            conn.close()

    def test_batch_search_same_results_as_individual(self):
        """Batch queries should produce same results as individual queries."""
        from rag_system.search.bm25_search import BM25Search
        from rag_system.database import get_connection, get_doc_terms, get_doc_terms_batch

        conn = get_connection(self.temp_path)
        try:
            # Get terms individually
            terms1 = get_doc_terms(conn, self.chunk1_id)
            terms2 = get_doc_terms(conn, self.chunk2_id)

            # Get terms in batch
            batch_result = get_doc_terms_batch(conn, [self.chunk1_id, self.chunk2_id])

            self.assertEqual(terms1, batch_result[self.chunk1_id])
            self.assertEqual(terms2, batch_result[self.chunk2_id])
        finally:
            conn.close()


class TestIncrementalIndex(unittest.TestCase):
    """Test incremental BM25 index updates."""

    def setUp(self):
        """Create temporary database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)
        self.page_id = insert_page(conn, url='http://test.com', title='Test',
                                   raw_html='', parsed_text='', content_hash='abc')
        self.chunk1_id = insert_chunk(conn, page_id=self.page_id, chunk_type='small',
                                      chunk_index=0, content='initial content',
                                      heading_path='')
        conn.close()

        # Build initial index
        from rag_system.search.bm25_search import BM25Index
        self.bm25_index = BM25Index(self.temp_path)
        self.bm25_index.build()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_index_chunk_adds_new_chunk(self):
        """index_chunk should add a new chunk to the index."""
        from rag_system.database import get_connection, insert_chunk, get_doc_terms

        conn = get_connection(self.temp_path)
        new_chunk_id = insert_chunk(conn, page_id=self.page_id, chunk_type='small',
                                    chunk_index=1, content='new python content',
                                    heading_path='')
        conn.close()

        self.bm25_index.index_chunk(new_chunk_id, 'new python content')

        # Verify terms were indexed
        conn = get_connection(self.temp_path)
        terms = get_doc_terms(conn, new_chunk_id)
        conn.close()

        self.assertIn('python', terms)
        self.assertIn('content', terms)

    def test_index_chunk_updates_existing(self):
        """index_chunk should update an existing chunk."""
        from rag_system.database import get_connection, get_doc_terms

        # Re-index with different content
        self.bm25_index.index_chunk(self.chunk1_id, 'updated python content')

        # Verify new terms
        conn = get_connection(self.temp_path)
        terms = get_doc_terms(conn, self.chunk1_id)
        conn.close()

        self.assertIn('python', terms)
        self.assertIn('updated', terms)
        self.assertNotIn('initial', terms)

    def test_remove_chunk(self):
        """remove_chunk should remove chunk from index."""
        from rag_system.database import get_connection, get_doc_terms

        self.bm25_index.remove_chunk(self.chunk1_id)

        # Verify terms were removed
        conn = get_connection(self.temp_path)
        terms = get_doc_terms(conn, self.chunk1_id)
        conn.close()

        self.assertEqual(len(terms), 0)

    def test_index_chunks_batch(self):
        """index_chunks_batch should index multiple chunks efficiently."""
        from rag_system.database import get_connection, insert_chunk, get_doc_terms, get_corpus_stats

        conn = get_connection(self.temp_path)
        chunk2_id = insert_chunk(conn, page_id=self.page_id, chunk_type='small',
                                 chunk_index=1, content='', heading_path='')
        chunk3_id = insert_chunk(conn, page_id=self.page_id, chunk_type='small',
                                 chunk_index=2, content='', heading_path='')
        conn.close()

        # Batch index
        self.bm25_index.index_chunks_batch([
            (chunk2_id, 'python programming'),
            (chunk3_id, 'java programming')
        ])

        # Verify terms
        conn = get_connection(self.temp_path)
        terms2 = get_doc_terms(conn, chunk2_id)
        terms3 = get_doc_terms(conn, chunk3_id)
        stats = get_corpus_stats(conn)
        conn.close()

        self.assertIn('python', terms2)
        self.assertIn('java', terms3)
        # Should have 3 total docs now (1 original + 2 new)
        self.assertEqual(stats['total_docs'], 3)


class TestBM25SearchBatchPerformance(unittest.TestCase):
    """Test that batch queries are used in search."""

    def setUp(self):
        """Create temporary database with many chunks."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='', content_hash='abc')

        # Create many chunks
        for i in range(20):
            insert_chunk(conn, page_id=page_id, chunk_type='small',
                        chunk_index=i, content=f'chunk {i} programming content test',
                        heading_path='')
        conn.close()

        from rag_system.search.bm25_search import BM25Index
        index = BM25Index(self.temp_path)
        index.build()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_search_many_chunks(self):
        """Search should work efficiently with many matching chunks."""
        from rag_system.search.bm25_search import BM25Search

        searcher = BM25Search(self.temp_path)
        results = searcher.search('programming', top_k=10)

        # Should get results
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 10)


if __name__ == '__main__':
    unittest.main()
