"""Tests for the indexer module."""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os


class TestIndexer(unittest.TestCase):
    """Test Indexer class."""

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

    def test_indexer_initialization(self):
        """Indexer should initialize with database path."""
        from rag_system.ingestion.indexer import Indexer

        indexer = Indexer(self.temp_path)

        self.assertEqual(indexer.db_path, self.temp_path)

    def test_index_page_stores_page(self):
        """index_page should store page in database."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, get_page_by_url

        indexer = Indexer(self.temp_path)

        page_data = {
            'url': 'https://example.com/test',
            'html': '<html><head><title>Test</title></head><body>Content</body></html>',
            'status_code': 200
        }

        page_id = indexer.index_page(page_data)

        self.assertIsNotNone(page_id)

        # Verify page is in database
        conn = get_connection(self.temp_path)
        page = get_page_by_url(conn, 'https://example.com/test')
        conn.close()

        self.assertIsNotNone(page)
        self.assertEqual(page['url'], 'https://example.com/test')

    def test_index_page_creates_chunks(self):
        """index_page should create chunks for the page."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, get_chunks_by_page

        indexer = Indexer(self.temp_path)

        # Page with enough content to create chunks
        page_data = {
            'url': 'https://example.com/test',
            'html': '<html><body>' + '<p>Test content paragraph.</p>' * 50 + '</body></html>',
            'status_code': 200
        }

        page_id = indexer.index_page(page_data)

        conn = get_connection(self.temp_path)
        chunks = get_chunks_by_page(conn, page_id)
        conn.close()

        self.assertGreater(len(chunks), 0)

    def test_index_page_skips_duplicate(self):
        """index_page should skip pages with same URL."""
        from rag_system.ingestion.indexer import Indexer

        indexer = Indexer(self.temp_path)

        page_data = {
            'url': 'https://example.com/test',
            'html': '<html><body>Content</body></html>',
            'status_code': 200
        }

        # Index twice
        page_id1 = indexer.index_page(page_data)
        page_id2 = indexer.index_page(page_data)

        # Second should return None (skipped) or same ID
        self.assertTrue(page_id2 is None or page_id2 == page_id1)

    def test_index_page_with_embeddings(self):
        """index_page should store embeddings when vector client provided."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, get_all_chunks_with_embeddings

        # Mock vector client
        mock_client = MagicMock()
        mock_client.get_embeddings_batch.return_value = [[0.1, 0.2, 0.3]]

        indexer = Indexer(self.temp_path, vector_client=mock_client)

        page_data = {
            'url': 'https://example.com/test',
            'html': '<html><body>Short content.</body></html>',
            'status_code': 200
        }

        indexer.index_page(page_data)

        conn = get_connection(self.temp_path)
        chunks_with_embeddings = get_all_chunks_with_embeddings(conn)
        conn.close()

        self.assertGreater(len(chunks_with_embeddings), 0)


class TestBatchIndexing(unittest.TestCase):
    """Test batch indexing functionality."""

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

    def test_index_pages_batch(self):
        """index_pages should index multiple pages."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, get_all_pages

        indexer = Indexer(self.temp_path)

        pages = [
            {'url': 'https://example.com/1', 'html': '<html><body>Page 1</body></html>', 'status_code': 200},
            {'url': 'https://example.com/2', 'html': '<html><body>Page 2</body></html>', 'status_code': 200},
        ]

        results = indexer.index_pages(pages)

        conn = get_connection(self.temp_path)
        all_pages = get_all_pages(conn)
        conn.close()

        self.assertEqual(len(all_pages), 2)


class TestCrawlAndIndex(unittest.TestCase):
    """Test combined crawl and index functionality."""

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

    def test_crawl_and_index(self):
        """crawl_and_index should crawl site and index pages."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.ingestion.crawler import Crawler
        from rag_system.database import get_connection, get_all_pages

        # Mock crawler
        mock_pages = [
            {'url': 'https://example.com/', 'html': '<html><body>Home</body></html>', 'status_code': 200},
            {'url': 'https://example.com/about', 'html': '<html><body>About</body></html>', 'status_code': 200},
        ]

        indexer = Indexer(self.temp_path)

        with patch.object(Crawler, 'crawl', return_value=iter(mock_pages)):
            stats = indexer.crawl_and_index(
                start_url='https://example.com',
                allowed_domains=['example.com'],
                max_pages=10
            )

        conn = get_connection(self.temp_path)
        all_pages = get_all_pages(conn)
        conn.close()

        self.assertEqual(len(all_pages), 2)
        self.assertEqual(stats['pages_indexed'], 2)

    def test_bm25_index_built_after_crawl(self):
        """crawl_and_index should build BM25 index after indexing pages."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.ingestion.crawler import Crawler
        from rag_system.database import get_connection

        mock_pages = [
            {'url': 'https://example.com/', 'html': '<html><body>Home page content</body></html>', 'status_code': 200},
            {'url': 'https://example.com/about', 'html': '<html><body>About page content</body></html>', 'status_code': 200},
        ]

        indexer = Indexer(self.temp_path)

        with patch.object(Crawler, 'crawl', return_value=iter(mock_pages)):
            stats = indexer.crawl_and_index(
                start_url='https://example.com',
                allowed_domains=['example.com'],
                max_pages=10
            )

        # Verify BM25 index tables are populated
        conn = get_connection(self.temp_path)

        # Check doc_terms table
        cursor = conn.execute("SELECT COUNT(*) as count FROM doc_terms")
        doc_terms_count = cursor.fetchone()['count']
        self.assertGreater(doc_terms_count, 0, "doc_terms table should have entries")

        # Check corpus_stats table
        cursor = conn.execute("SELECT COUNT(*) as count FROM corpus_stats")
        corpus_stats_count = cursor.fetchone()['count']
        self.assertEqual(corpus_stats_count, 1, "corpus_stats should have one row")

        # Check term_doc_frequencies table
        cursor = conn.execute("SELECT COUNT(*) as count FROM term_doc_frequencies")
        term_freq_count = cursor.fetchone()['count']
        self.assertGreater(term_freq_count, 0, "term_doc_frequencies should have unique terms")

        conn.close()


class TestIndexStats(unittest.TestCase):
    """Test index statistics functionality."""

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

    def test_get_stats(self):
        """get_stats should return index statistics."""
        from rag_system.ingestion.indexer import Indexer

        indexer = Indexer(self.temp_path)

        # Index a page
        page_data = {
            'url': 'https://example.com/test',
            'html': '<html><body>' + '<p>Content paragraph.</p>' * 20 + '</body></html>',
            'status_code': 200
        }
        indexer.index_page(page_data)

        stats = indexer.get_stats()

        self.assertIn('total_pages', stats)
        self.assertIn('total_chunks', stats)
        self.assertEqual(stats['total_pages'], 1)
        self.assertGreater(stats['total_chunks'], 0)


class TestCrawlSessionIntegration(unittest.TestCase):
    """Test crawl session integration with indexer."""

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

    def test_crawl_and_index_creates_session(self):
        """crawl_and_index should create a crawl session."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, list_crawl_sessions

        indexer = Indexer(self.temp_path)

        # Mock crawler to return one page
        def mock_crawl_generator():
            yield {
                'url': 'https://example.com',
                'html': '<html><body>Test</body></html>',
                'status_code': 200
            }

        with patch('rag_system.ingestion.indexer.Crawler') as MockCrawler:
            mock_crawler = MagicMock()
            mock_crawler.crawl.return_value = mock_crawl_generator()
            mock_crawler.queue = []
            mock_crawler.visited = {'https://example.com'}
            MockCrawler.return_value = mock_crawler

            indexer.crawl_and_index(
                start_url='https://example.com',
                allowed_domains=['example.com'],
                max_pages=10
            )

        conn = get_connection(self.temp_path)
        sessions = list_crawl_sessions(conn)
        conn.close()

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]['start_url'], 'https://example.com')

    def test_crawl_and_index_marks_session_completed(self):
        """crawl_and_index should mark session as completed when done."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, list_crawl_sessions

        indexer = Indexer(self.temp_path)

        def mock_crawl_generator():
            yield {
                'url': 'https://example.com',
                'html': '<html><body>Test</body></html>',
                'status_code': 200
            }

        with patch('rag_system.ingestion.indexer.Crawler') as MockCrawler:
            mock_crawler = MagicMock()
            mock_crawler.crawl.return_value = mock_crawl_generator()
            mock_crawler.queue = []
            mock_crawler.visited = {'https://example.com'}
            MockCrawler.return_value = mock_crawler

            indexer.crawl_and_index(
                start_url='https://example.com',
                allowed_domains=['example.com'],
                max_pages=10
            )

        conn = get_connection(self.temp_path)
        sessions = list_crawl_sessions(conn)
        conn.close()

        self.assertEqual(sessions[0]['status'], 'completed')

    def test_crawl_and_index_resumes_interrupted_session(self):
        """crawl_and_index should resume an interrupted session."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import (
            get_connection, create_crawl_session, update_session_status,
            add_urls_to_crawl_queue, get_crawl_session
        )

        # Create an interrupted session with URLs in queue
        conn = get_connection(self.temp_path)
        session_id = create_crawl_session(
            conn, 'https://example.com', ['example.com'], 100
        )
        update_session_status(conn, session_id, 'interrupted')
        add_urls_to_crawl_queue(conn, session_id, [
            'https://example.com/page1',
            'https://example.com/page2'
        ])
        conn.close()

        indexer = Indexer(self.temp_path)

        pages_yielded = []

        def mock_crawl_generator():
            for url in ['https://example.com/page1', 'https://example.com/page2']:
                pages_yielded.append(url)
                yield {
                    'url': url,
                    'html': '<html><body>Test</body></html>',
                    'status_code': 200
                }

        with patch('rag_system.ingestion.indexer.Crawler') as MockCrawler:
            mock_crawler = MagicMock()
            mock_crawler.crawl.return_value = mock_crawl_generator()
            mock_crawler.queue = []
            mock_crawler.visited = set()
            MockCrawler.return_value = mock_crawler

            result = indexer.crawl_and_index(
                start_url='https://example.com',
                allowed_domains=['example.com'],
                max_pages=100
            )

        # Should have resumed the existing session
        conn = get_connection(self.temp_path)
        session = get_crawl_session(conn, session_id)
        conn.close()

        self.assertEqual(session['status'], 'completed')

    def test_crawl_and_index_fresh_ignores_existing_session(self):
        """crawl_and_index with fresh=True should start a new session."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import (
            get_connection, create_crawl_session, update_session_status,
            list_crawl_sessions
        )

        # Create an interrupted session
        conn = get_connection(self.temp_path)
        session_id = create_crawl_session(
            conn, 'https://example.com', ['example.com'], 100
        )
        update_session_status(conn, session_id, 'interrupted')
        conn.close()

        indexer = Indexer(self.temp_path)

        def mock_crawl_generator():
            yield {
                'url': 'https://example.com',
                'html': '<html><body>Test</body></html>',
                'status_code': 200
            }

        with patch('rag_system.ingestion.indexer.Crawler') as MockCrawler:
            mock_crawler = MagicMock()
            mock_crawler.crawl.return_value = mock_crawl_generator()
            mock_crawler.queue = []
            mock_crawler.visited = {'https://example.com'}
            MockCrawler.return_value = mock_crawler

            indexer.crawl_and_index(
                start_url='https://example.com',
                allowed_domains=['example.com'],
                max_pages=10,
                fresh=True
            )

        conn = get_connection(self.temp_path)
        sessions = list_crawl_sessions(conn)
        conn.close()

        # Should have two sessions (old interrupted + new completed)
        self.assertEqual(len(sessions), 2)
        # Most recent should be completed
        self.assertEqual(sessions[0]['status'], 'completed')

    def test_crawl_and_index_updates_session_stats(self):
        """crawl_and_index should update session statistics."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, list_crawl_sessions

        indexer = Indexer(self.temp_path)

        def mock_crawl_generator():
            for i in range(3):
                yield {
                    'url': f'https://example.com/page{i}',
                    'html': '<html><body>Test content</body></html>',
                    'status_code': 200
                }

        with patch('rag_system.ingestion.indexer.Crawler') as MockCrawler:
            mock_crawler = MagicMock()
            mock_crawler.crawl.return_value = mock_crawl_generator()
            mock_crawler.queue = []
            mock_crawler.visited = set()
            MockCrawler.return_value = mock_crawler

            indexer.crawl_and_index(
                start_url='https://example.com',
                allowed_domains=['example.com'],
                max_pages=10
            )

        conn = get_connection(self.temp_path)
        sessions = list_crawl_sessions(conn)
        conn.close()

        self.assertEqual(sessions[0]['pages_crawled'], 3)
        self.assertGreaterEqual(sessions[0]['pages_indexed'], 0)

    def test_crawl_and_index_saves_queue_on_interrupt(self):
        """crawl_and_index should save queue when interrupted."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import (
            get_connection, list_crawl_sessions, get_crawl_queue_urls
        )

        indexer = Indexer(self.temp_path)

        # Simulate an interrupted crawl
        def mock_crawl_generator():
            yield {
                'url': 'https://example.com',
                'html': '<html><body>Test</body></html>',
                'status_code': 200
            }
            # Simulate interrupt after first page
            raise KeyboardInterrupt()

        with patch('rag_system.ingestion.indexer.Crawler') as MockCrawler:
            mock_crawler = MagicMock()
            mock_crawler.crawl.return_value = mock_crawl_generator()
            mock_crawler.queue = ['https://example.com/page1', 'https://example.com/page2']
            mock_crawler.visited = {'https://example.com'}
            MockCrawler.return_value = mock_crawler

            try:
                indexer.crawl_and_index(
                    start_url='https://example.com',
                    allowed_domains=['example.com'],
                    max_pages=10
                )
            except KeyboardInterrupt:
                pass  # Expected

        conn = get_connection(self.temp_path)
        sessions = list_crawl_sessions(conn)
        # Queue should be saved
        queue_urls = get_crawl_queue_urls(conn, sessions[0]['id'])
        conn.close()

        self.assertEqual(sessions[0]['status'], 'interrupted')
        self.assertEqual(len(queue_urls), 2)


if __name__ == '__main__':
    unittest.main()
