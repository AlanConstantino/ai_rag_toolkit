"""Tests for the BM25 search module."""

import unittest
import tempfile
import os


class TestPorterStemmer(unittest.TestCase):
    """Test Porter Stemmer implementation."""

    def test_stem_basic_words(self):
        """PorterStemmer should stem common word forms."""
        from rag_system.search.bm25_search import PorterStemmer

        stemmer = PorterStemmer()

        # Test plural forms
        self.assertEqual(stemmer.stem('cats'), 'cat')
        self.assertEqual(stemmer.stem('ponies'), 'poni')

        # Test -ing forms
        self.assertEqual(stemmer.stem('running'), 'run')
        self.assertEqual(stemmer.stem('walking'), 'walk')

        # Test -ed forms
        self.assertEqual(stemmer.stem('walked'), 'walk')
        self.assertEqual(stemmer.stem('agreed'), 'agre')

    def test_stem_preserves_short_words(self):
        """PorterStemmer should preserve very short words."""
        from rag_system.search.bm25_search import PorterStemmer

        stemmer = PorterStemmer()

        self.assertEqual(stemmer.stem('a'), 'a')
        self.assertEqual(stemmer.stem('an'), 'an')
        self.assertEqual(stemmer.stem('be'), 'be')

    def test_stem_caches_results(self):
        """PorterStemmer should cache results for performance."""
        from rag_system.search.bm25_search import PorterStemmer

        stemmer = PorterStemmer()

        # First call
        result1 = stemmer.stem('programming')
        # Second call (should use cache)
        result2 = stemmer.stem('programming')

        self.assertEqual(result1, result2)
        self.assertIn('programming', stemmer._cache)

    def test_stem_common_suffixes(self):
        """PorterStemmer should handle common suffixes."""
        from rag_system.search.bm25_search import PorterStemmer

        stemmer = PorterStemmer()

        # -ational -> -ate
        self.assertEqual(stemmer.stem('relational'), 'relat')

        # -iveness -> -ive
        self.assertEqual(stemmer.stem('effectiveness'), 'effect')

        # -ization -> -ize
        self.assertEqual(stemmer.stem('organization'), 'organ')


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

    def test_tokenize_splits_camel_case(self):
        """tokenize_for_bm25 should split camelCase words."""
        from rag_system.search.bm25_search import tokenize_for_bm25

        result = tokenize_for_bm25("getUserName processDataRequest")

        self.assertIn('get', result)
        self.assertIn('user', result)
        self.assertIn('name', result)
        self.assertIn('process', result)
        self.assertIn('data', result)
        self.assertIn('request', result)

    def test_tokenize_splits_snake_case(self):
        """tokenize_for_bm25 should split snake_case words."""
        from rag_system.search.bm25_search import tokenize_for_bm25

        result = tokenize_for_bm25("get_user_name process_data")

        self.assertIn('get', result)
        self.assertIn('user', result)
        self.assertIn('name', result)
        self.assertIn('process', result)
        self.assertIn('data', result)

    def test_tokenize_splits_kebab_case(self):
        """tokenize_for_bm25 should split kebab-case words."""
        from rag_system.search.bm25_search import tokenize_for_bm25

        result = tokenize_for_bm25("get-user-name data-processing")

        self.assertIn('get', result)
        self.assertIn('user', result)
        self.assertIn('name', result)
        self.assertIn('data', result)
        self.assertIn('processing', result)

    def test_tokenize_with_stemming(self):
        """tokenize_for_bm25 should apply stemming when enabled."""
        from rag_system.search.bm25_search import tokenize_for_bm25

        result = tokenize_for_bm25("running cats programming", stem=True)

        self.assertIn('run', result)
        self.assertIn('cat', result)
        self.assertIn('program', result)

    def test_tokenize_without_stemming(self):
        """tokenize_for_bm25 should preserve word forms when stemming disabled."""
        from rag_system.search.bm25_search import tokenize_for_bm25

        result = tokenize_for_bm25("running cats programming", stem=False)

        self.assertIn('running', result)
        self.assertIn('cats', result)
        self.assertIn('programming', result)


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
            # Terms are stemmed by default but "python" and "java" don't change
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
            # Terms are stemmed: "programming" -> "program"
            result = get_term_doc_frequencies_batch(conn, ['python', 'program', 'nonexistent'])

            self.assertEqual(result['python'], 1)
            self.assertEqual(result['program'], 2)  # stemmed from "programming"
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

        # Verify new terms (stemmed: "updated" -> "updat")
        conn = get_connection(self.temp_path)
        terms = get_doc_terms(conn, self.chunk1_id)
        conn.close()

        self.assertIn('python', terms)
        self.assertIn('updat', terms)  # stemmed from "updated"
        self.assertNotIn('initi', terms)  # stemmed from "initial"

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


class TestBM25SearchWithStemming(unittest.TestCase):
    """Test BM25 search with stemming enabled."""

    def setUp(self):
        """Create temporary database with test data."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='', content_hash='abc')
        # Use words with different forms
        self.chunk1_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=0, content='running quickly through forests',
                                      heading_path='')
        self.chunk2_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=1, content='she runs fast in the forest',
                                      heading_path='')
        self.chunk3_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=2, content='database connections available',
                                      heading_path='')
        conn.close()

        # Build index with stemming
        from rag_system.search.bm25_search import BM25Index
        index = BM25Index(self.temp_path, use_stemming=True)
        index.build()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_stemming_matches_word_variants(self):
        """Search should match different word forms via stemming."""
        from rag_system.search.bm25_search import BM25Search

        searcher = BM25Search(self.temp_path)
        # Search for "run" should match "running" and "runs"
        results = searcher.search('run', top_k=10, stem=True)

        chunk_ids = [r[0] for r in results]
        self.assertIn(self.chunk1_id, chunk_ids)  # has "running"
        self.assertIn(self.chunk2_id, chunk_ids)  # has "runs"
        self.assertNotIn(self.chunk3_id, chunk_ids)  # no run-related words

    def test_stemming_matches_plural_singular(self):
        """Search should match singular and plural forms."""
        from rag_system.search.bm25_search import BM25Search

        searcher = BM25Search(self.temp_path)
        # Search for "forest" should match "forests" and "forest"
        results = searcher.search('forest', top_k=10, stem=True)

        chunk_ids = [r[0] for r in results]
        self.assertIn(self.chunk1_id, chunk_ids)  # has "forests"
        self.assertIn(self.chunk2_id, chunk_ids)  # has "forest"


class TestBM25SearchWithQueryExpansion(unittest.TestCase):
    """Test BM25 search with query expansion."""

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
                                      chunk_index=0, content='configure the system settings',
                                      heading_path='')
        self.chunk2_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=1, content='setup guide for new users',
                                      heading_path='')
        self.chunk3_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=2, content='database optimization tips',
                                      heading_path='')
        conn.close()

        # Build index with stemming
        from rag_system.search.bm25_search import BM25Index
        index = BM25Index(self.temp_path, use_stemming=True)
        index.build()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_search_with_query_expander(self):
        """Search with query expansion should find more results via synonyms."""
        from rag_system.search.bm25_search import BM25Search
        from rag_system.query.expander import QueryExpander

        expander = QueryExpander()
        searcher = BM25Search(self.temp_path, query_expander=expander)

        # Search for "configure" - expander should also try "setup"
        results = searcher.search('configure', top_k=10, expand_query=True, stem=True)

        chunk_ids = [r[0] for r in results]
        # Should find chunk1 (has "configure") and chunk2 (has "setup" which is synonym)
        self.assertIn(self.chunk1_id, chunk_ids)
        self.assertIn(self.chunk2_id, chunk_ids)

    def test_search_without_query_expander(self):
        """Search without expander should only match exact terms."""
        from rag_system.search.bm25_search import BM25Search

        searcher = BM25Search(self.temp_path)

        # Search for "configure" without expansion - should only match chunk1
        results = searcher.search('configure', top_k=10, expand_query=False, stem=True)

        chunk_ids = [r[0] for r in results]
        self.assertIn(self.chunk1_id, chunk_ids)
        # chunk2 with "setup" should NOT be found without expansion
        self.assertNotIn(self.chunk2_id, chunk_ids)

    def test_expansion_takes_max_score(self):
        """Query expansion should use max score when chunk matches multiple queries."""
        from rag_system.search.bm25_search import BM25Search
        from rag_system.query.expander import QueryExpander

        expander = QueryExpander()
        searcher = BM25Search(self.temp_path, query_expander=expander)

        # Search with expansion
        results = searcher.search('configure', top_k=10, expand_query=True, stem=True)

        # Verify we get results (not duplicates)
        chunk_ids = [r[0] for r in results]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))  # No duplicates


class TestBM25IndexWithStemming(unittest.TestCase):
    """Test BM25 index with stemming configuration."""

    def setUp(self):
        """Create temporary database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_index_with_stemming_stores_stems(self):
        """BM25Index with stemming should store stemmed terms."""
        from rag_system.search.bm25_search import BM25Index
        from rag_system.database import get_connection, insert_page, insert_chunk, get_doc_terms

        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='', content_hash='abc')
        chunk_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                               chunk_index=0, content='running programs quickly',
                               heading_path='')
        conn.close()

        index = BM25Index(self.temp_path, use_stemming=True)
        index.build()

        conn = get_connection(self.temp_path)
        terms = get_doc_terms(conn, chunk_id)
        conn.close()

        # Should have stemmed versions
        self.assertIn('run', terms)  # stemmed from "running"
        self.assertIn('program', terms)  # stemmed from "programs"
        self.assertIn('quickli', terms)  # stemmed from "quickly"

    def test_index_without_stemming_stores_original(self):
        """BM25Index without stemming should store original terms."""
        from rag_system.search.bm25_search import BM25Index
        from rag_system.database import get_connection, insert_page, insert_chunk, get_doc_terms

        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='', content_hash='abc')
        chunk_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                               chunk_index=0, content='running programs quickly',
                               heading_path='')
        conn.close()

        index = BM25Index(self.temp_path, use_stemming=False)
        index.build()

        conn = get_connection(self.temp_path)
        terms = get_doc_terms(conn, chunk_id)
        conn.close()

        # Should have original forms
        self.assertIn('running', terms)
        self.assertIn('programs', terms)
        self.assertIn('quickly', terms)


if __name__ == '__main__':
    unittest.main()
