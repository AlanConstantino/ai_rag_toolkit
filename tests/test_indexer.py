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


if __name__ == '__main__':
    unittest.main()
