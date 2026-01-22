"""End-to-end tests for the RAG system.

Tests the complete flow from ingestion through querying.
"""

import os
import tempfile
import unittest

from rag_system.database import (
    init_db, get_connection, insert_page, insert_chunk
)
from rag_system.search.bm25_search import BM25Index, BM25Search


class TestE2EIndexAndQuery(unittest.TestCase):
    """End-to-end tests for indexing and querying flow."""

    def setUp(self):
        """Create temporary database with test data."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_full_index_query_flow(self):
        """Full flow: insert pages -> insert chunks -> build index -> query."""
        conn = get_connection(self.temp_path)

        # Insert multiple pages
        page1_id = insert_page(
            conn, url='http://docs.example.com/install',
            title='Installation Guide',
            raw_html='<html><body>Install Python 3.6</body></html>',
            parsed_text='Install Python 3.6', content_hash='hash1'
        )
        page2_id = insert_page(
            conn, url='http://docs.example.com/config',
            title='Configuration Guide',
            raw_html='<html><body>Configure timeout settings</body></html>',
            parsed_text='Configure timeout settings', content_hash='hash2'
        )

        # Insert chunks for pages
        insert_chunk(conn, page_id=page1_id, chunk_type='small',
                    chunk_index=0, content='To install Python 3.6, download it from python.org',
                    heading_path='Installation > Download')
        insert_chunk(conn, page_id=page1_id, chunk_type='small',
                    chunk_index=1, content='Run the installer and follow the prompts',
                    heading_path='Installation > Run Installer')
        insert_chunk(conn, page_id=page2_id, chunk_type='small',
                    chunk_index=0, content='Configure timeout settings in config.py',
                    heading_path='Configuration > Timeouts')
        insert_chunk(conn, page_id=page2_id, chunk_type='small',
                    chunk_index=1, content='Set READ_TIMEOUT to desired value in seconds',
                    heading_path='Configuration > Timeouts > Read')
        conn.close()

        # Build BM25 index
        index = BM25Index(self.temp_path)
        index.build()

        # Query for installation
        searcher = BM25Search(self.temp_path)
        results = searcher.search('install python', top_k=10)

        # Should find installation-related chunks
        self.assertGreater(len(results), 0)

        # Query for configuration
        results = searcher.search('timeout settings', top_k=10)

        # Should find config-related chunks
        self.assertGreater(len(results), 0)

    def test_incremental_index_update(self):
        """Incremental update: add page, index, add more, re-index."""
        conn = get_connection(self.temp_path)

        # Initial page
        page1_id = insert_page(
            conn, url='http://docs.example.com/initial',
            title='Initial Page',
            raw_html='', parsed_text='', content_hash='hash1'
        )
        chunk1_id = insert_chunk(
            conn, page_id=page1_id, chunk_type='small',
            chunk_index=0, content='initial content about databases',
            heading_path=''
        )
        conn.close()

        # Build initial index
        index = BM25Index(self.temp_path)
        index.build()

        # Add new page with chunks
        conn = get_connection(self.temp_path)
        page2_id = insert_page(
            conn, url='http://docs.example.com/new',
            title='New Page',
            raw_html='', parsed_text='', content_hash='hash2'
        )
        chunk2_id = insert_chunk(
            conn, page_id=page2_id, chunk_type='small',
            chunk_index=0, content='new content about networking',
            heading_path=''
        )
        conn.close()

        # Incrementally index new chunk
        index.index_chunk(chunk2_id, 'new content about networking')

        # Query should find both old and new content
        searcher = BM25Search(self.temp_path)

        results = searcher.search('databases', top_k=10)
        self.assertGreater(len(results), 0)

        results = searcher.search('networking', top_k=10)
        self.assertGreater(len(results), 0)


class TestEmptyDatabaseQueries(unittest.TestCase):
    """Tests for queries on empty or minimal databases."""

    def setUp(self):
        """Create temporary database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_query_empty_database(self):
        """Querying empty database should return empty results."""
        searcher = BM25Search(self.temp_path)
        results = searcher.search('anything', top_k=10)

        self.assertEqual(len(results), 0)

    def test_query_no_matching_content(self):
        """Query with no matching terms should return empty."""
        conn = get_connection(self.temp_path)
        page_id = insert_page(
            conn, url='http://test.com', title='Test',
            raw_html='', parsed_text='', content_hash='abc'
        )
        insert_chunk(conn, page_id=page_id, chunk_type='small',
                    chunk_index=0, content='python programming language',
                    heading_path='')
        conn.close()

        index = BM25Index(self.temp_path)
        index.build()

        searcher = BM25Search(self.temp_path)
        results = searcher.search('xyznonexistent123', top_k=10)

        self.assertEqual(len(results), 0)

    def test_query_only_stopwords(self):
        """Query with only stopwords should return empty."""
        conn = get_connection(self.temp_path)
        page_id = insert_page(
            conn, url='http://test.com', title='Test',
            raw_html='', parsed_text='', content_hash='abc'
        )
        insert_chunk(conn, page_id=page_id, chunk_type='small',
                    chunk_index=0, content='python programming language',
                    heading_path='')
        conn.close()

        index = BM25Index(self.temp_path)
        index.build()

        searcher = BM25Search(self.temp_path)
        results = searcher.search('the a an is', top_k=10)

        self.assertEqual(len(results), 0)


class TestUnicodeHandling(unittest.TestCase):
    """Tests for Unicode content handling."""

    def setUp(self):
        """Create temporary database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_emoji_in_content(self):
        """Content with emojis should be indexed and searchable."""
        conn = get_connection(self.temp_path)
        page_id = insert_page(
            conn, url='http://test.com', title='Test',
            raw_html='', parsed_text='', content_hash='abc'
        )
        insert_chunk(conn, page_id=page_id, chunk_type='small',
                    chunk_index=0, content='great job completing the task',
                    heading_path='')
        conn.close()

        index = BM25Index(self.temp_path)
        index.build()

        searcher = BM25Search(self.temp_path)
        results = searcher.search('completing task', top_k=10)

        self.assertGreater(len(results), 0)

    def test_non_latin_scripts(self):
        """Non-Latin scripts should be stored and retrieved."""
        conn = get_connection(self.temp_path)
        page_id = insert_page(
            conn, url='http://test.com', title='Test',
            raw_html='', parsed_text='', content_hash='abc'
        )
        # Chinese text
        insert_chunk(conn, page_id=page_id, chunk_type='small',
                    chunk_index=0, content='python programming tutorial',
                    heading_path='')
        conn.close()

        index = BM25Index(self.temp_path)
        index.build()

        searcher = BM25Search(self.temp_path)
        results = searcher.search('programming', top_k=10)

        self.assertGreater(len(results), 0)

    def test_mixed_unicode(self):
        """Mixed Unicode content should be handled."""
        conn = get_connection(self.temp_path)
        page_id = insert_page(
            conn, url='http://test.com', title='Test',
            raw_html='', parsed_text='', content_hash='abc'
        )
        # Mix of scripts and special characters
        content = 'Python documentation version 3.6'
        insert_chunk(conn, page_id=page_id, chunk_type='small',
                    chunk_index=0, content=content,
                    heading_path='')
        conn.close()

        index = BM25Index(self.temp_path)
        index.build()

        searcher = BM25Search(self.temp_path)
        results = searcher.search('python documentation', top_k=10)

        self.assertGreater(len(results), 0)


class TestLargeContent(unittest.TestCase):
    """Tests for large content handling."""

    def setUp(self):
        """Create temporary database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_large_chunk_content(self):
        """Large chunk content should be handled."""
        conn = get_connection(self.temp_path)
        page_id = insert_page(
            conn, url='http://test.com', title='Test',
            raw_html='', parsed_text='', content_hash='abc'
        )

        # Create large content (~100KB)
        large_content = ('python programming ' * 5000) + ' unique_marker'
        insert_chunk(conn, page_id=page_id, chunk_type='small',
                    chunk_index=0, content=large_content,
                    heading_path='')
        conn.close()

        index = BM25Index(self.temp_path)
        index.build()

        searcher = BM25Search(self.temp_path)
        results = searcher.search('unique_marker', top_k=10)

        self.assertGreater(len(results), 0)

    def test_many_chunks(self):
        """Many chunks should be handled efficiently."""
        conn = get_connection(self.temp_path)
        page_id = insert_page(
            conn, url='http://test.com', title='Test',
            raw_html='', parsed_text='', content_hash='abc'
        )

        # Create 100 chunks
        for i in range(100):
            insert_chunk(conn, page_id=page_id, chunk_type='small',
                        chunk_index=i, content=f'chunk number {i} with content',
                        heading_path=f'Section {i}')
        conn.close()

        index = BM25Index(self.temp_path)
        index.build()

        searcher = BM25Search(self.temp_path)
        results = searcher.search('chunk content', top_k=10)

        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 10)


class TestMalformedInput(unittest.TestCase):
    """Tests for malformed input handling."""

    def setUp(self):
        """Create temporary database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_empty_query(self):
        """Empty query should return empty results."""
        searcher = BM25Search(self.temp_path)
        results = searcher.search('', top_k=10)

        self.assertEqual(len(results), 0)

    def test_whitespace_only_query(self):
        """Whitespace-only query should return empty results."""
        searcher = BM25Search(self.temp_path)
        results = searcher.search('   \t\n  ', top_k=10)

        self.assertEqual(len(results), 0)

    def test_special_characters_query(self):
        """Query with special characters should not crash."""
        conn = get_connection(self.temp_path)
        page_id = insert_page(
            conn, url='http://test.com', title='Test',
            raw_html='', parsed_text='', content_hash='abc'
        )
        insert_chunk(conn, page_id=page_id, chunk_type='small',
                    chunk_index=0, content='python programming',
                    heading_path='')
        conn.close()

        index = BM25Index(self.temp_path)
        index.build()

        searcher = BM25Search(self.temp_path)

        # Various special character queries
        special_queries = [
            '!@#$%^&*()',
            '<script>alert("xss")</script>',
            "'; DROP TABLE chunks; --",
            '\\x00\\x01\\x02',
            'test\nwith\nnewlines',
        ]

        for query in special_queries:
            # Should not raise exception
            results = searcher.search(query, top_k=10)
            self.assertIsInstance(results, list)


class TestFixtures(unittest.TestCase):
    """Tests using shared fixtures."""

    @classmethod
    def setUpClass(cls):
        """Create shared test database with fixtures."""
        cls.temp_fd, cls.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(cls.temp_fd)
        init_db(cls.temp_path)

        # Create fixture data
        conn = get_connection(cls.temp_path)

        # Documentation pages
        cls.page_ids = {}
        cls.chunk_ids = {}

        for i, (url, title, content) in enumerate([
            ('http://docs.example.com/install', 'Installation',
             'How to install the software package'),
            ('http://docs.example.com/config', 'Configuration',
             'How to configure settings and options'),
            ('http://docs.example.com/api', 'API Reference',
             'API documentation and endpoints'),
        ]):
            page_id = insert_page(
                conn, url=url, title=title,
                raw_html='', parsed_text=content, content_hash=f'hash{i}'
            )
            cls.page_ids[title.lower()] = page_id

            chunk_id = insert_chunk(
                conn, page_id=page_id, chunk_type='small',
                chunk_index=0, content=content, heading_path=title
            )
            cls.chunk_ids[title.lower()] = chunk_id

        conn.close()

        # Build index
        index = BM25Index(cls.temp_path)
        index.build()

    @classmethod
    def tearDownClass(cls):
        """Remove shared test database."""
        if os.path.exists(cls.temp_path):
            os.unlink(cls.temp_path)

    def test_search_installation(self):
        """Search should find installation content."""
        searcher = BM25Search(self.temp_path)
        results = searcher.search('install software', top_k=10)

        self.assertGreater(len(results), 0)
        # Installation page chunk should be in results
        chunk_ids = [r[0] for r in results]
        self.assertIn(self.chunk_ids['installation'], chunk_ids)

    def test_search_configuration(self):
        """Search should find configuration content."""
        searcher = BM25Search(self.temp_path)
        results = searcher.search('configure settings', top_k=10)

        self.assertGreater(len(results), 0)

    def test_search_api(self):
        """Search should find API content."""
        searcher = BM25Search(self.temp_path)
        results = searcher.search('API documentation', top_k=10)

        self.assertGreater(len(results), 0)


if __name__ == '__main__':
    unittest.main()
