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


class TestEmbeddingStats(unittest.TestCase):
    """Test EmbeddingStats class."""

    def test_embedding_stats_initialization(self):
        """EmbeddingStats should initialize with zero values."""
        from rag_system.ingestion.indexer import EmbeddingStats

        stats = EmbeddingStats()

        self.assertEqual(stats.chunks_total, 0)
        self.assertEqual(stats.chunks_embedded, 0)
        self.assertEqual(stats.chunks_skipped, 0)
        self.assertEqual(stats.chunks_failed, 0)
        self.assertFalse(stats.rate_limit_hit)
        self.assertFalse(stats.interrupted)
        self.assertIsNone(stats.error_message)

    def test_embedding_stats_to_dict(self):
        """EmbeddingStats.to_dict should return all values."""
        from rag_system.ingestion.indexer import EmbeddingStats

        stats = EmbeddingStats()
        stats.chunks_total = 100
        stats.chunks_embedded = 80
        stats.chunks_skipped = 10
        stats.chunks_failed = 10
        stats.rate_limit_hit = True
        stats.interrupted = False
        stats.error_message = 'Test error'

        result = stats.to_dict()

        self.assertEqual(result['chunks_total'], 100)
        self.assertEqual(result['chunks_embedded'], 80)
        self.assertEqual(result['chunks_skipped'], 10)
        self.assertEqual(result['chunks_failed'], 10)
        self.assertTrue(result['rate_limit_hit'])
        self.assertFalse(result['interrupted'])
        self.assertEqual(result['error_message'], 'Test error')


class TestResumeEmbeddings(unittest.TestCase):
    """Test resume_embeddings functionality."""

    def setUp(self):
        """Create temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk
        init_db(self.temp_path)

        # Create a page with some chunks
        conn = get_connection(self.temp_path)
        self.page_id = insert_page(
            conn, 'https://example.com/test', 'Test Page', '', 'Test content', 'abc123'
        )
        self.chunk_ids = []
        for i in range(3):
            chunk_id = insert_chunk(
                conn, self.page_id, 'small', i, f'Chunk {i} content', f'Section {i}'
            )
            self.chunk_ids.append(chunk_id)
        conn.close()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_resume_embeddings_without_vector_client(self):
        """resume_embeddings should return error when no vector client."""
        from rag_system.ingestion.indexer import Indexer

        indexer = Indexer(self.temp_path)

        result = indexer.resume_embeddings()

        self.assertIn('error', result)
        self.assertEqual(result['error'], 'No vector client configured')

    def test_resume_embeddings_no_chunks_to_embed(self):
        """resume_embeddings should return early when all chunks have embeddings."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, update_chunk_embedding

        # Add embeddings to all chunks
        conn = get_connection(self.temp_path)
        for chunk_id in self.chunk_ids:
            update_chunk_embedding(conn, chunk_id, [0.1, 0.2, 0.3])
        conn.close()

        mock_client = MagicMock()
        indexer = Indexer(self.temp_path, vector_client=mock_client)

        result = indexer.resume_embeddings()

        self.assertEqual(result['chunks_total'], 0)
        self.assertEqual(result['chunks_embedded'], 0)
        self.assertTrue(result['completed'])
        mock_client.get_embeddings_batch.assert_not_called()

    def test_resume_embeddings_embeds_missing_chunks(self):
        """resume_embeddings should embed chunks without embeddings."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, get_all_chunks_with_embeddings

        mock_client = MagicMock()
        mock_client.get_embeddings_batch.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        indexer = Indexer(self.temp_path, vector_client=mock_client)

        result = indexer.resume_embeddings()

        self.assertEqual(result['chunks_total'], 3)
        self.assertEqual(result['chunks_embedded'], 3)
        self.assertTrue(result['completed'])

        # Verify embeddings were stored
        conn = get_connection(self.temp_path)
        chunks_with_embeddings = get_all_chunks_with_embeddings(conn)
        conn.close()

        self.assertEqual(len(chunks_with_embeddings), 3)

    def test_resume_embeddings_filters_by_page_id(self):
        """resume_embeddings should filter by page_id when specified."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, insert_page, insert_chunk

        # Create another page with chunks
        conn = get_connection(self.temp_path)
        page2_id = insert_page(
            conn, 'https://example.com/page2', 'Page 2', '', 'Content', 'def456'
        )
        insert_chunk(conn, page2_id, 'small', 0, 'Page 2 chunk', 'Section A')
        conn.close()

        mock_client = MagicMock()
        mock_client.get_embeddings_batch.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        indexer = Indexer(self.temp_path, vector_client=mock_client)

        # Only embed chunks from first page
        result = indexer.resume_embeddings(page_id=self.page_id)

        self.assertEqual(result['chunks_total'], 3)  # Only first page's chunks

    def test_resume_embeddings_creates_job(self):
        """resume_embeddings should create an embedding job for tracking."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, list_embedding_jobs

        mock_client = MagicMock()
        mock_client.get_embeddings_batch.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        indexer = Indexer(self.temp_path, vector_client=mock_client)

        result = indexer.resume_embeddings()

        self.assertIn('job_id', result)

        # Verify job was created
        conn = get_connection(self.temp_path)
        jobs = list_embedding_jobs(conn)
        conn.close()

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['status'], 'completed')


class TestGenerateEmbeddingsWithRateLimit(unittest.TestCase):
    """Test _generate_embeddings rate limit handling."""

    def setUp(self):
        """Create temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)
        self.page_id = insert_page(
            conn, 'https://example.com/test', 'Test Page', '', 'Test content', 'abc123'
        )
        self.chunk_ids = []
        for i in range(2):
            chunk_id = insert_chunk(
                conn, self.page_id, 'small', i, f'Chunk {i} content', f'Section {i}'
            )
            self.chunk_ids.append(chunk_id)
        conn.close()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_generate_embeddings_raises_rate_limit_error(self):
        """_generate_embeddings should raise RateLimitError when STOP_ON_RATE_LIMIT=True."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.api_client import APIError, RateLimitError
        from rag_system.database import get_connection

        mock_client = MagicMock()
        mock_client.get_embeddings_batch.side_effect = APIError(
            "Rate limited", status_code=429
        )

        indexer = Indexer(self.temp_path, vector_client=mock_client)

        conn = get_connection(self.temp_path)
        chunks = [
            {'content': 'Chunk 0 content', 'heading_path': 'Section 0'},
            {'content': 'Chunk 1 content', 'heading_path': 'Section 1'}
        ]

        with patch('rag_system.ingestion.indexer.config') as mock_config:
            mock_config.EMBEDDING_BATCH_SIZE = 100
            mock_config.EMBEDDING_BATCH_DELAY = 0
            mock_config.EMBEDDING_MAX_RETRIES = 2
            mock_config.EMBEDDING_RETRY_DELAY = 0.01
            mock_config.EMBEDDING_STOP_ON_RATE_LIMIT = True

            with patch('time.sleep'):  # Skip delays
                with self.assertRaises(RateLimitError):
                    indexer._generate_embeddings(conn, chunks, self.chunk_ids, 'Test Page')

        conn.close()

    def test_generate_embeddings_returns_stats_on_success(self):
        """_generate_embeddings should return EmbeddingStats on success."""
        from rag_system.ingestion.indexer import Indexer, EmbeddingStats
        from rag_system.database import get_connection

        mock_client = MagicMock()
        mock_client.get_embeddings_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]

        indexer = Indexer(self.temp_path, vector_client=mock_client)

        conn = get_connection(self.temp_path)
        chunks = [
            {'content': 'Chunk 0 content', 'heading_path': 'Section 0'},
            {'content': 'Chunk 1 content', 'heading_path': 'Section 1'}
        ]

        stats = indexer._generate_embeddings(conn, chunks, self.chunk_ids, 'Test Page')

        self.assertIsInstance(stats, EmbeddingStats)
        self.assertEqual(stats.chunks_total, 2)
        self.assertEqual(stats.chunks_embedded, 2)
        self.assertEqual(stats.chunks_failed, 0)
        self.assertFalse(stats.rate_limit_hit)

        conn.close()


class TestEntityExtractionIntegration(unittest.TestCase):
    """Tests for entity extraction integration in indexer pipeline."""

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

    def test_entity_extraction_called_when_enabled(self):
        """Entity extraction should be called when enabled and chat client available."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection

        mock_chat_client = MagicMock()
        mock_chat_client.complete_json.return_value = {
            'entities': [
                {'name': 'TestSystem', 'type': 'system', 'description': 'A test system'}
            ],
            'relationships': []
        }

        indexer = Indexer(self.temp_path, chat_client=mock_chat_client)

        page_data = {
            'url': 'https://example.com/test',
            'html': '<html><head><title>Test</title></head><body><p>TestSystem is great.</p></body></html>',
            'status_code': 200
        }

        with patch('rag_system.ingestion.indexer.config') as mock_config:
            mock_config.ENTITY_EXTRACTION_ENABLED = True
            mock_config.PAGE_SUMMARIZATION_ENABLED = False
            mock_config.SMALL_CHUNK_SIZE = 500
            mock_config.LARGE_CHUNK_SIZE = 2000
            mock_config.CHUNK_OVERLAP = 100

            page_id = indexer.index_page(page_data)

        self.assertIsNotNone(page_id)
        # Verify entity extraction was called
        mock_chat_client.complete_json.assert_called()

    def test_entity_extraction_skipped_when_disabled(self):
        """Entity extraction should not be called when disabled."""
        from rag_system.ingestion.indexer import Indexer

        mock_chat_client = MagicMock()
        indexer = Indexer(self.temp_path, chat_client=mock_chat_client)

        page_data = {
            'url': 'https://example.com/test',
            'html': '<html><head><title>Test</title></head><body>Content</body></html>',
            'status_code': 200
        }

        with patch('rag_system.ingestion.indexer.config') as mock_config:
            mock_config.ENTITY_EXTRACTION_ENABLED = False
            mock_config.PAGE_SUMMARIZATION_ENABLED = False
            mock_config.SMALL_CHUNK_SIZE = 500
            mock_config.LARGE_CHUNK_SIZE = 2000
            mock_config.CHUNK_OVERLAP = 100

            page_id = indexer.index_page(page_data)

        self.assertIsNotNone(page_id)
        # complete_json is used for entity extraction, should not be called
        mock_chat_client.complete_json.assert_not_called()

    def test_entities_stored_in_database(self):
        """Extracted entities should be stored in the database."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, get_entity_by_name

        mock_chat_client = MagicMock()
        mock_chat_client.complete_json.return_value = {
            'entities': [
                {'name': 'ConfigManager', 'type': 'system', 'description': 'Manages configs'}
            ],
            'relationships': []
        }

        indexer = Indexer(self.temp_path, chat_client=mock_chat_client)

        page_data = {
            'url': 'https://example.com/config',
            'html': '<html><head><title>Config</title></head><body><p>ConfigManager handles all settings.</p></body></html>',
            'status_code': 200
        }

        with patch('rag_system.ingestion.indexer.config') as mock_config:
            mock_config.ENTITY_EXTRACTION_ENABLED = True
            mock_config.PAGE_SUMMARIZATION_ENABLED = False
            mock_config.SMALL_CHUNK_SIZE = 500
            mock_config.LARGE_CHUNK_SIZE = 2000
            mock_config.CHUNK_OVERLAP = 100

            indexer.index_page(page_data)

        # Verify entity is in database
        conn = get_connection(self.temp_path)
        try:
            entity = get_entity_by_name(conn, 'ConfigManager')
            self.assertIsNotNone(entity)
            self.assertEqual(entity['type'], 'system')
        finally:
            conn.close()

    def test_entity_extraction_failure_does_not_block_indexing(self):
        """Entity extraction failures should not prevent page indexing."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, get_page_by_url

        mock_chat_client = MagicMock()
        mock_chat_client.complete_json.side_effect = Exception("LLM API error")

        indexer = Indexer(self.temp_path, chat_client=mock_chat_client)

        page_data = {
            'url': 'https://example.com/test',
            'html': '<html><head><title>Test</title></head><body>Content</body></html>',
            'status_code': 200
        }

        with patch('rag_system.ingestion.indexer.config') as mock_config:
            mock_config.ENTITY_EXTRACTION_ENABLED = True
            mock_config.PAGE_SUMMARIZATION_ENABLED = False
            mock_config.SMALL_CHUNK_SIZE = 500
            mock_config.LARGE_CHUNK_SIZE = 2000
            mock_config.CHUNK_OVERLAP = 100

            # Should not raise, despite extraction failure
            page_id = indexer.index_page(page_data)

        self.assertIsNotNone(page_id)

        # Page should still be indexed
        conn = get_connection(self.temp_path)
        try:
            page = get_page_by_url(conn, 'https://example.com/test')
            self.assertIsNotNone(page)
        finally:
            conn.close()


class TestPageSummarizationIntegration(unittest.TestCase):
    """Tests for page summarization integration in indexer pipeline."""

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

    def test_summarization_called_when_enabled(self):
        """Page summarization should be called when enabled."""
        from rag_system.ingestion.indexer import Indexer

        mock_chat_client = MagicMock()
        mock_chat_client.complete.return_value = "This page is about configuration."

        indexer = Indexer(self.temp_path, chat_client=mock_chat_client)

        page_data = {
            'url': 'https://example.com/config',
            'html': '<html><head><title>Config</title></head><body><p>Configuration details here.</p></body></html>',
            'status_code': 200
        }

        with patch('rag_system.ingestion.indexer.config') as mock_config:
            mock_config.ENTITY_EXTRACTION_ENABLED = False
            mock_config.PAGE_SUMMARIZATION_ENABLED = True
            mock_config.SMALL_CHUNK_SIZE = 500
            mock_config.LARGE_CHUNK_SIZE = 2000
            mock_config.CHUNK_OVERLAP = 100

            indexer.index_page(page_data)

        # complete is used for summarization
        mock_chat_client.complete.assert_called()

    def test_summary_stored_in_database(self):
        """Page summary should be stored in the pages table."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, get_page_by_url

        mock_chat_client = MagicMock()
        mock_chat_client.complete.return_value = "A summary of the page content."

        indexer = Indexer(self.temp_path, chat_client=mock_chat_client)

        page_data = {
            'url': 'https://example.com/test',
            'html': '<html><head><title>Test</title></head><body><p>Page content here.</p></body></html>',
            'status_code': 200
        }

        with patch('rag_system.ingestion.indexer.config') as mock_config:
            mock_config.ENTITY_EXTRACTION_ENABLED = False
            mock_config.PAGE_SUMMARIZATION_ENABLED = True
            mock_config.SMALL_CHUNK_SIZE = 500
            mock_config.LARGE_CHUNK_SIZE = 2000
            mock_config.CHUNK_OVERLAP = 100

            indexer.index_page(page_data)

        # Verify summary is stored
        conn = get_connection(self.temp_path)
        try:
            page = get_page_by_url(conn, 'https://example.com/test')
            self.assertIsNotNone(page)
            self.assertEqual(page['summary'], "A summary of the page content.")
        finally:
            conn.close()

    def test_summarization_skipped_when_disabled(self):
        """Summarization should not be called when disabled."""
        from rag_system.ingestion.indexer import Indexer

        mock_chat_client = MagicMock()
        indexer = Indexer(self.temp_path, chat_client=mock_chat_client)

        page_data = {
            'url': 'https://example.com/test',
            'html': '<html><head><title>Test</title></head><body>Content</body></html>',
            'status_code': 200
        }

        with patch('rag_system.ingestion.indexer.config') as mock_config:
            mock_config.ENTITY_EXTRACTION_ENABLED = False
            mock_config.PAGE_SUMMARIZATION_ENABLED = False
            mock_config.SMALL_CHUNK_SIZE = 500
            mock_config.LARGE_CHUNK_SIZE = 2000
            mock_config.CHUNK_OVERLAP = 100

            indexer.index_page(page_data)

        # complete should not be called for summarization
        mock_chat_client.complete.assert_not_called()

    def test_summarization_failure_does_not_block_indexing(self):
        """Summarization failures should not prevent page indexing."""
        from rag_system.ingestion.indexer import Indexer
        from rag_system.database import get_connection, get_page_by_url

        mock_chat_client = MagicMock()
        mock_chat_client.complete.side_effect = Exception("LLM API error")

        indexer = Indexer(self.temp_path, chat_client=mock_chat_client)

        page_data = {
            'url': 'https://example.com/test',
            'html': '<html><head><title>Test</title></head><body>Content</body></html>',
            'status_code': 200
        }

        with patch('rag_system.ingestion.indexer.config') as mock_config:
            mock_config.ENTITY_EXTRACTION_ENABLED = False
            mock_config.PAGE_SUMMARIZATION_ENABLED = True
            mock_config.SMALL_CHUNK_SIZE = 500
            mock_config.LARGE_CHUNK_SIZE = 2000
            mock_config.CHUNK_OVERLAP = 100

            page_id = indexer.index_page(page_data)

        self.assertIsNotNone(page_id)

        # Page should still be indexed
        conn = get_connection(self.temp_path)
        try:
            page = get_page_by_url(conn, 'https://example.com/test')
            self.assertIsNotNone(page)
        finally:
            conn.close()


if __name__ == '__main__':
    unittest.main()
