"""Tests for the database module."""

import unittest
import os
import tempfile
import sqlite3


class TestDatabaseInit(unittest.TestCase):
    """Test database initialization."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_init_db_creates_tables(self):
        """init_db should create all required tables."""
        from rag_system.database import init_db
        init_db(self.temp_path)

        conn = sqlite3.connect(self.temp_path)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {
            'pages', 'chunks', 'entities', 'relationships',
            'chunk_entities', 'systems', 'page_systems', 'global_summary',
            'doc_terms', 'corpus_stats', 'term_doc_frequencies',
            'query_cache', 'query_log',
            'crawl_sessions', 'crawl_queue'
        }

        self.assertTrue(expected_tables.issubset(tables),
                       f"Missing tables: {expected_tables - tables}")
        conn.close()

    def test_init_db_creates_indexes(self):
        """init_db should create required indexes."""
        from rag_system.database import init_db
        init_db(self.temp_path)

        conn = sqlite3.connect(self.temp_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cursor.fetchall()}

        expected_indexes = {
            'idx_chunks_page', 'idx_chunks_parent',
            'idx_entities_name', 'idx_entities_type',
            'idx_doc_terms_term', 'idx_doc_terms_chunk'
        }

        self.assertTrue(expected_indexes.issubset(indexes),
                       f"Missing indexes: {expected_indexes - indexes}")
        conn.close()

    def test_init_db_is_idempotent(self):
        """init_db should be safe to call multiple times."""
        from rag_system.database import init_db
        init_db(self.temp_path)
        init_db(self.temp_path)  # Should not raise

        conn = sqlite3.connect(self.temp_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        self.assertGreater(len(tables), 0)
        conn.close()


class TestDatabaseConnection(unittest.TestCase):
    """Test database connection helpers."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_get_connection_returns_connection(self):
        """get_connection should return a sqlite3 connection."""
        from rag_system.database import get_connection
        conn = get_connection(self.temp_path)
        self.assertIsInstance(conn, sqlite3.Connection)
        conn.close()

    def test_get_connection_enables_foreign_keys(self):
        """get_connection should enable foreign key support."""
        from rag_system.database import get_connection
        conn = get_connection(self.temp_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        result = cursor.fetchone()[0]
        self.assertEqual(result, 1)
        conn.close()


class TestPageOperations(unittest.TestCase):
    """Test page CRUD operations."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_insert_page(self):
        """insert_page should insert a page and return its id."""
        from rag_system.database import get_connection, insert_page
        conn = get_connection(self.temp_path)

        page_id = insert_page(
            conn,
            url='https://example.com/page1',
            title='Test Page',
            raw_html='<html><body>Test</body></html>',
            parsed_text='Test',
            content_hash='abc123'
        )

        self.assertIsInstance(page_id, int)
        self.assertGreater(page_id, 0)
        conn.close()

    def test_get_page_by_url(self):
        """get_page_by_url should retrieve a page by its URL."""
        from rag_system.database import get_connection, insert_page, get_page_by_url
        conn = get_connection(self.temp_path)

        url = 'https://example.com/page1'
        insert_page(conn, url=url, title='Test Page', raw_html='<html></html>',
                   parsed_text='Test', content_hash='abc123')

        page = get_page_by_url(conn, url)

        self.assertIsNotNone(page)
        self.assertEqual(page['url'], url)
        self.assertEqual(page['title'], 'Test Page')
        conn.close()

    def test_get_page_by_url_returns_none_for_missing(self):
        """get_page_by_url should return None for non-existent URL."""
        from rag_system.database import get_connection, get_page_by_url
        conn = get_connection(self.temp_path)

        page = get_page_by_url(conn, 'https://nonexistent.com')

        self.assertIsNone(page)
        conn.close()

    def test_update_page_summary(self):
        """update_page_summary should update the summary field."""
        from rag_system.database import get_connection, insert_page, update_page_summary, get_page_by_url
        conn = get_connection(self.temp_path)

        url = 'https://example.com/page1'
        page_id = insert_page(conn, url=url, title='Test', raw_html='',
                             parsed_text='', content_hash='abc')

        update_page_summary(conn, page_id, 'This is a summary.')

        page = get_page_by_url(conn, url)
        self.assertEqual(page['summary'], 'This is a summary.')
        conn.close()

    def test_get_all_pages(self):
        """get_all_pages should return all pages."""
        from rag_system.database import get_connection, insert_page, get_all_pages
        conn = get_connection(self.temp_path)

        insert_page(conn, url='https://example.com/1', title='Page 1',
                   raw_html='', parsed_text='', content_hash='a')
        insert_page(conn, url='https://example.com/2', title='Page 2',
                   raw_html='', parsed_text='', content_hash='b')

        pages = get_all_pages(conn)

        self.assertEqual(len(pages), 2)
        conn.close()


class TestChunkOperations(unittest.TestCase):
    """Test chunk CRUD operations."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page
        init_db(self.temp_path)
        conn = get_connection(self.temp_path)
        self.page_id = insert_page(conn, url='https://example.com',
                                   title='Test', raw_html='', parsed_text='',
                                   content_hash='abc')
        conn.close()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_insert_chunk(self):
        """insert_chunk should insert a chunk and return its id."""
        from rag_system.database import get_connection, insert_chunk
        conn = get_connection(self.temp_path)

        chunk_id = insert_chunk(
            conn,
            page_id=self.page_id,
            chunk_type='large',
            chunk_index=0,
            content='This is chunk content.',
            heading_path='Section > Subsection'
        )

        self.assertIsInstance(chunk_id, int)
        self.assertGreater(chunk_id, 0)
        conn.close()

    def test_insert_chunk_with_parent(self):
        """insert_chunk should support parent_chunk_id for small chunks."""
        from rag_system.database import get_connection, insert_chunk
        conn = get_connection(self.temp_path)

        parent_id = insert_chunk(conn, page_id=self.page_id, chunk_type='large',
                                chunk_index=0, content='Parent content',
                                heading_path='Section')

        child_id = insert_chunk(conn, page_id=self.page_id, chunk_type='small',
                               chunk_index=0, content='Child content',
                               heading_path='Section', parent_chunk_id=parent_id)

        self.assertIsInstance(child_id, int)
        self.assertNotEqual(parent_id, child_id)
        conn.close()

    def test_get_chunks_by_page(self):
        """get_chunks_by_page should return all chunks for a page."""
        from rag_system.database import get_connection, insert_chunk, get_chunks_by_page
        conn = get_connection(self.temp_path)

        insert_chunk(conn, page_id=self.page_id, chunk_type='large',
                    chunk_index=0, content='Chunk 1', heading_path='A')
        insert_chunk(conn, page_id=self.page_id, chunk_type='large',
                    chunk_index=1, content='Chunk 2', heading_path='B')

        chunks = get_chunks_by_page(conn, self.page_id)

        self.assertEqual(len(chunks), 2)
        conn.close()

    def test_update_chunk_embedding(self):
        """update_chunk_embedding should store embedding JSON."""
        from rag_system.database import get_connection, insert_chunk, update_chunk_embedding, get_chunk_by_id
        import json
        conn = get_connection(self.temp_path)

        chunk_id = insert_chunk(conn, page_id=self.page_id, chunk_type='large',
                               chunk_index=0, content='Test', heading_path='A')

        embedding = [0.1, 0.2, 0.3]
        update_chunk_embedding(conn, chunk_id, embedding)

        chunk = get_chunk_by_id(conn, chunk_id)
        stored_embedding = json.loads(chunk['embedding_json'])

        self.assertEqual(stored_embedding, embedding)
        conn.close()

    def test_get_all_chunks_with_embeddings(self):
        """get_all_chunks_with_embeddings should return chunks that have embeddings."""
        from rag_system.database import get_connection, insert_chunk, update_chunk_embedding, get_all_chunks_with_embeddings
        conn = get_connection(self.temp_path)

        chunk1_id = insert_chunk(conn, page_id=self.page_id, chunk_type='small',
                                chunk_index=0, content='With embedding', heading_path='A')
        chunk2_id = insert_chunk(conn, page_id=self.page_id, chunk_type='small',
                                chunk_index=1, content='No embedding', heading_path='B')

        update_chunk_embedding(conn, chunk1_id, [0.1, 0.2])

        chunks = get_all_chunks_with_embeddings(conn)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['id'], chunk1_id)
        conn.close()


class TestEntityOperations(unittest.TestCase):
    """Test entity CRUD operations."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page
        init_db(self.temp_path)
        conn = get_connection(self.temp_path)
        self.page_id = insert_page(conn, url='https://example.com',
                                   title='Test', raw_html='', parsed_text='',
                                   content_hash='abc')
        conn.close()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_insert_entity(self):
        """insert_entity should insert an entity and return its id."""
        from rag_system.database import get_connection, insert_entity
        conn = get_connection(self.temp_path)

        entity_id = insert_entity(
            conn,
            name='Redis',
            entity_type='system',
            description='In-memory data store',
            page_id=self.page_id
        )

        self.assertIsInstance(entity_id, int)
        self.assertGreater(entity_id, 0)
        conn.close()

    def test_get_entity_by_name(self):
        """get_entity_by_name should find entity by name."""
        from rag_system.database import get_connection, insert_entity, get_entity_by_name
        conn = get_connection(self.temp_path)

        insert_entity(conn, name='Redis', entity_type='system',
                     description='Cache', page_id=self.page_id)

        entity = get_entity_by_name(conn, 'Redis')

        self.assertIsNotNone(entity)
        self.assertEqual(entity['name'], 'Redis')
        conn.close()

    def test_insert_relationship(self):
        """insert_relationship should create a relationship between entities."""
        from rag_system.database import get_connection, insert_entity, insert_relationship
        conn = get_connection(self.temp_path)

        entity1_id = insert_entity(conn, name='App', entity_type='system',
                                  description='Application', page_id=self.page_id)
        entity2_id = insert_entity(conn, name='Redis', entity_type='system',
                                  description='Cache', page_id=self.page_id)

        rel_id = insert_relationship(
            conn,
            source_entity_id=entity1_id,
            target_entity_id=entity2_id,
            relationship_type='depends_on',
            description='App uses Redis for caching'
        )

        self.assertIsInstance(rel_id, int)
        conn.close()

    def test_link_chunk_to_entity(self):
        """link_chunk_to_entity should create a chunk-entity association."""
        from rag_system.database import get_connection, insert_entity, insert_chunk, link_chunk_to_entity, get_entities_for_chunk
        conn = get_connection(self.temp_path)

        entity_id = insert_entity(conn, name='Redis', entity_type='system',
                                 description='Cache', page_id=self.page_id)
        chunk_id = insert_chunk(conn, page_id=self.page_id, chunk_type='large',
                               chunk_index=0, content='Redis content', heading_path='A')

        link_chunk_to_entity(conn, chunk_id, entity_id)

        entities = get_entities_for_chunk(conn, chunk_id)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['name'], 'Redis')
        conn.close()


class TestBM25Operations(unittest.TestCase):
    """Test BM25 index operations."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk
        init_db(self.temp_path)
        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='https://example.com',
                             title='Test', raw_html='', parsed_text='',
                             content_hash='abc')
        self.chunk_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                    chunk_index=0, content='test content',
                                    heading_path='A')
        conn.close()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_insert_doc_terms(self):
        """insert_doc_terms should store term frequencies."""
        from rag_system.database import get_connection, insert_doc_terms, get_doc_terms
        conn = get_connection(self.temp_path)

        terms = {'hello': 2, 'world': 1}
        insert_doc_terms(conn, self.chunk_id, terms)

        stored_terms = get_doc_terms(conn, self.chunk_id)
        self.assertEqual(stored_terms['hello'], 2)
        self.assertEqual(stored_terms['world'], 1)
        conn.close()

    def test_update_corpus_stats(self):
        """update_corpus_stats should store corpus statistics."""
        from rag_system.database import get_connection, update_corpus_stats, get_corpus_stats
        conn = get_connection(self.temp_path)

        update_corpus_stats(conn, total_docs=100, avg_doc_length=150.5)

        stats = get_corpus_stats(conn)
        self.assertEqual(stats['total_docs'], 100)
        self.assertAlmostEqual(stats['avg_doc_length'], 150.5)
        conn.close()

    def test_update_term_doc_frequencies(self):
        """update_term_doc_frequencies should store document frequencies."""
        from rag_system.database import get_connection, update_term_doc_frequencies, get_term_doc_frequency
        conn = get_connection(self.temp_path)

        term_freqs = {'hello': 10, 'world': 5}
        update_term_doc_frequencies(conn, term_freqs)

        self.assertEqual(get_term_doc_frequency(conn, 'hello'), 10)
        self.assertEqual(get_term_doc_frequency(conn, 'world'), 5)
        self.assertEqual(get_term_doc_frequency(conn, 'nonexistent'), 0)
        conn.close()


class TestSummaryOperations(unittest.TestCase):
    """Test summary operations."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_set_global_summary(self):
        """set_global_summary should store the global summary."""
        from rag_system.database import get_connection, set_global_summary, get_global_summary
        conn = get_connection(self.temp_path)

        set_global_summary(conn, 'This is the global summary.')

        summary = get_global_summary(conn)
        self.assertEqual(summary, 'This is the global summary.')
        conn.close()

    def test_set_global_summary_replaces_existing(self):
        """set_global_summary should replace existing summary."""
        from rag_system.database import get_connection, set_global_summary, get_global_summary
        conn = get_connection(self.temp_path)

        set_global_summary(conn, 'First summary')
        set_global_summary(conn, 'Second summary')

        summary = get_global_summary(conn)
        self.assertEqual(summary, 'Second summary')
        conn.close()

    def test_insert_system(self):
        """insert_system should create a system entry."""
        from rag_system.database import get_connection, insert_system, get_system_by_name
        conn = get_connection(self.temp_path)

        system_id = insert_system(conn, name='Authentication',
                                 description='Handles user auth',
                                 summary='Auth system summary')

        system = get_system_by_name(conn, 'Authentication')
        self.assertIsNotNone(system)
        self.assertEqual(system['name'], 'Authentication')
        conn.close()


class TestQueryCacheOperations(unittest.TestCase):
    """Test query cache operations."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_cache_query(self):
        """cache_query should store query information."""
        from rag_system.database import get_connection, cache_query, get_cached_query
        import json
        conn = get_connection(self.temp_path)

        cache_query(
            conn,
            query_hash='abc123',
            query_type='factual',
            expanded_queries=['how to X', 'what is X'],
            embedding=[0.1, 0.2, 0.3]
        )

        cached = get_cached_query(conn, 'abc123')
        self.assertIsNotNone(cached)
        self.assertEqual(cached['query_type'], 'factual')
        self.assertEqual(json.loads(cached['expanded_queries']), ['how to X', 'what is X'])
        conn.close()

    def test_log_query(self):
        """log_query should record query execution details."""
        from rag_system.database import get_connection, log_query, get_query_logs
        conn = get_connection(self.temp_path)

        log_query(
            conn,
            query='test query',
            query_type='factual',
            expanded_queries=['expanded'],
            retrieved_chunk_ids=[1, 2, 3],
            confidence_score=0.85,
            answer_generated=True
        )

        logs = get_query_logs(conn, limit=10)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]['query'], 'test query')
        conn.close()


class TestDatabaseErrorHandling(unittest.TestCase):
    """Test database error handling functionality."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_insert_duplicate_page_raises_integrity_error(self):
        """insert_page should raise IntegrityConstraintError for duplicate URL."""
        from rag_system.database import (
            get_connection, insert_page, IntegrityConstraintError
        )
        conn = get_connection(self.temp_path)

        # Insert first page
        insert_page(conn, url='https://example.com/page1', title='Test',
                   raw_html='', parsed_text='', content_hash='abc')

        # Try to insert duplicate
        with self.assertRaises(IntegrityConstraintError):
            insert_page(conn, url='https://example.com/page1', title='Test 2',
                       raw_html='', parsed_text='', content_hash='def')

        conn.close()

    def test_insert_chunk_with_invalid_page_id_raises_error(self):
        """insert_chunk should raise IntegrityConstraintError for invalid page_id."""
        from rag_system.database import (
            get_connection, insert_chunk, IntegrityConstraintError
        )
        conn = get_connection(self.temp_path)

        with self.assertRaises(IntegrityConstraintError):
            insert_chunk(conn, page_id=99999, chunk_type='large',
                        chunk_index=0, content='Test', heading_path='A')

        conn.close()

    def test_transaction_context_manager_commits_on_success(self):
        """transaction() should commit when block completes successfully."""
        from rag_system.database import (
            get_connection, insert_page, get_page_by_url, transaction
        )
        conn = get_connection(self.temp_path)

        with transaction(conn):
            insert_page(conn, url='https://example.com/tx', title='TX Test',
                       raw_html='', parsed_text='', content_hash='abc',
                       auto_commit=False)

        # Verify the page was committed
        page = get_page_by_url(conn, 'https://example.com/tx')
        self.assertIsNotNone(page)
        conn.close()

    def test_transaction_context_manager_rolls_back_on_error(self):
        """transaction() should rollback when an error occurs."""
        from rag_system.database import (
            get_connection, insert_page, get_page_by_url, transaction,
            TransactionError
        )
        conn = get_connection(self.temp_path)

        try:
            with transaction(conn):
                insert_page(conn, url='https://example.com/rollback', title='Test',
                           raw_html='', parsed_text='', content_hash='abc',
                           auto_commit=False)
                raise ValueError("Simulated error")
        except TransactionError:
            pass

        # Verify the page was NOT committed
        page = get_page_by_url(conn, 'https://example.com/rollback')
        self.assertIsNone(page)
        conn.close()

    def test_transaction_rolls_back_on_integrity_error(self):
        """transaction() should rollback and raise on integrity error."""
        from rag_system.database import (
            get_connection, insert_page, get_page_by_url, transaction,
            IntegrityConstraintError
        )
        conn = get_connection(self.temp_path)

        # Insert first page
        insert_page(conn, url='https://example.com/first', title='First',
                   raw_html='', parsed_text='', content_hash='abc')

        try:
            with transaction(conn):
                # This should fail due to duplicate URL
                conn.execute(
                    "INSERT INTO pages (url, title, raw_html, parsed_text, content_hash) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ('https://example.com/first', 'Duplicate', '', '', 'def')
                )
        except IntegrityConstraintError:
            pass

        conn.close()

    def test_custom_exceptions_are_proper_subclasses(self):
        """Custom exceptions should be proper exception subclasses."""
        from rag_system.database import (
            DatabaseError, ConnectionError, TransactionError,
            IntegrityConstraintError, QueryError
        )

        self.assertTrue(issubclass(ConnectionError, DatabaseError))
        self.assertTrue(issubclass(TransactionError, DatabaseError))
        self.assertTrue(issubclass(IntegrityConstraintError, DatabaseError))
        self.assertTrue(issubclass(QueryError, DatabaseError))
        self.assertTrue(issubclass(DatabaseError, Exception))


class TestRetryDecorator(unittest.TestCase):
    """Test the retry_on_error decorator."""

    def test_retry_decorator_succeeds_on_first_try(self):
        """Decorator should not retry if operation succeeds."""
        from rag_system.database import retry_on_error

        call_count = [0]

        @retry_on_error(max_retries=3, retry_delay=0.01)
        def successful_operation():
            call_count[0] += 1
            return "success"

        result = successful_operation()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 1)

    def test_retry_decorator_retries_on_operational_error(self):
        """Decorator should retry on OperationalError."""
        from rag_system.database import retry_on_error, ConnectionError

        call_count = [0]

        @retry_on_error(max_retries=2, retry_delay=0.01)
        def flaky_operation():
            call_count[0] += 1
            if call_count[0] < 2:
                raise sqlite3.OperationalError("database is locked")
            return "success"

        result = flaky_operation()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 2)

    def test_retry_decorator_raises_after_max_retries(self):
        """Decorator should raise ConnectionError after max retries."""
        from rag_system.database import retry_on_error, ConnectionError

        call_count = [0]

        @retry_on_error(max_retries=2, retry_delay=0.01)
        def always_fails():
            call_count[0] += 1
            raise sqlite3.OperationalError("database is locked")

        with self.assertRaises(ConnectionError):
            always_fails()

        self.assertEqual(call_count[0], 3)  # Initial try + 2 retries


class TestAutoCommitParameter(unittest.TestCase):
    """Test the auto_commit parameter on database operations."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_auto_commit_false_does_not_commit(self):
        """Operations with auto_commit=False should not commit."""
        from rag_system.database import get_connection, insert_page, get_page_by_url
        conn = get_connection(self.temp_path)

        insert_page(conn, url='https://example.com/nocommit', title='Test',
                   raw_html='', parsed_text='', content_hash='abc',
                   auto_commit=False)

        # Rollback to verify it wasn't committed
        conn.rollback()

        page = get_page_by_url(conn, 'https://example.com/nocommit')
        self.assertIsNone(page)
        conn.close()

    def test_auto_commit_true_commits(self):
        """Operations with auto_commit=True (default) should commit."""
        from rag_system.database import get_connection, insert_page, get_page_by_url
        conn = get_connection(self.temp_path)

        insert_page(conn, url='https://example.com/commit', title='Test',
                   raw_html='', parsed_text='', content_hash='abc',
                   auto_commit=True)

        # Rollback should not affect committed data
        conn.rollback()

        page = get_page_by_url(conn, 'https://example.com/commit')
        self.assertIsNotNone(page)
        conn.close()


class TestCrawlSessionSchema(unittest.TestCase):
    """Test crawl session schema creation."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_init_db_creates_crawl_session_tables(self):
        """init_db should create crawl_sessions and crawl_queue tables."""
        from rag_system.database import init_db

        init_db(self.temp_path)

        conn = sqlite3.connect(self.temp_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        self.assertIn('crawl_sessions', tables)
        self.assertIn('crawl_queue', tables)
        conn.close()

    def test_crawl_sessions_table_has_required_columns(self):
        """crawl_sessions table should have all required columns."""
        from rag_system.database import init_db

        init_db(self.temp_path)

        conn = sqlite3.connect(self.temp_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(crawl_sessions)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            'id', 'start_url', 'allowed_domains', 'status',
            'pages_crawled', 'pages_indexed', 'pages_skipped', 'errors',
            'started_at', 'completed_at'
        }
        self.assertTrue(expected_columns.issubset(columns),
                       f"Missing columns: {expected_columns - columns}")
        conn.close()

    def test_crawl_queue_table_has_required_columns(self):
        """crawl_queue table should have all required columns."""
        from rag_system.database import init_db

        init_db(self.temp_path)

        conn = sqlite3.connect(self.temp_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(crawl_queue)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {'id', 'session_id', 'url', 'depth', 'added_at'}
        self.assertTrue(expected_columns.issubset(columns),
                       f"Missing columns: {expected_columns - columns}")
        conn.close()


class TestCrawlSessionOperations(unittest.TestCase):
    """Test crawl session CRUD operations."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db
        init_db(self.temp_path)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_create_crawl_session(self):
        """create_crawl_session should create a new session and return its ID."""
        from rag_system.database import get_connection, create_crawl_session

        conn = get_connection(self.temp_path)
        session_id = create_crawl_session(
            conn,
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=100
        )

        self.assertIsInstance(session_id, int)
        self.assertGreater(session_id, 0)
        conn.close()

    def test_get_crawl_session(self):
        """get_crawl_session should retrieve session by ID."""
        from rag_system.database import get_connection, create_crawl_session, get_crawl_session

        conn = get_connection(self.temp_path)
        session_id = create_crawl_session(
            conn,
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=100
        )

        session = get_crawl_session(conn, session_id)

        self.assertIsNotNone(session)
        self.assertEqual(session['start_url'], 'https://example.com')
        self.assertEqual(session['status'], 'active')
        self.assertEqual(session['pages_crawled'], 0)
        conn.close()

    def test_get_crawl_session_not_found(self):
        """get_crawl_session should return None for non-existent session."""
        from rag_system.database import get_connection, get_crawl_session

        conn = get_connection(self.temp_path)
        session = get_crawl_session(conn, 9999)

        self.assertIsNone(session)
        conn.close()

    def test_get_active_session_for_url(self):
        """get_active_session_for_url should find resumable session."""
        from rag_system.database import (
            get_connection, create_crawl_session,
            update_session_status, get_active_session_for_url
        )

        conn = get_connection(self.temp_path)

        # Create a session and mark it interrupted
        session_id = create_crawl_session(
            conn,
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=100
        )
        update_session_status(conn, session_id, 'interrupted')

        # Should find the interrupted session
        found = get_active_session_for_url(conn, 'https://example.com')

        self.assertIsNotNone(found)
        self.assertEqual(found['id'], session_id)
        conn.close()

    def test_get_active_session_for_url_no_match(self):
        """get_active_session_for_url should return None when no session exists."""
        from rag_system.database import get_connection, get_active_session_for_url

        conn = get_connection(self.temp_path)
        found = get_active_session_for_url(conn, 'https://nonexistent.com')

        self.assertIsNone(found)
        conn.close()

    def test_get_active_session_ignores_completed(self):
        """get_active_session_for_url should not return completed sessions."""
        from rag_system.database import (
            get_connection, create_crawl_session,
            update_session_status, get_active_session_for_url
        )

        conn = get_connection(self.temp_path)

        session_id = create_crawl_session(
            conn,
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=100
        )
        update_session_status(conn, session_id, 'completed')

        found = get_active_session_for_url(conn, 'https://example.com')

        self.assertIsNone(found)
        conn.close()

    def test_update_session_status(self):
        """update_session_status should change session status."""
        from rag_system.database import (
            get_connection, create_crawl_session,
            update_session_status, get_crawl_session
        )

        conn = get_connection(self.temp_path)
        session_id = create_crawl_session(
            conn,
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=100
        )

        update_session_status(conn, session_id, 'completed')

        session = get_crawl_session(conn, session_id)
        self.assertEqual(session['status'], 'completed')
        conn.close()

    def test_update_session_stats(self):
        """update_session_stats should update crawl statistics."""
        from rag_system.database import (
            get_connection, create_crawl_session,
            update_session_stats, get_crawl_session
        )

        conn = get_connection(self.temp_path)
        session_id = create_crawl_session(
            conn,
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=100
        )

        update_session_stats(
            conn, session_id,
            pages_crawled=50,
            pages_indexed=45,
            pages_skipped=5,
            errors=2
        )

        session = get_crawl_session(conn, session_id)
        self.assertEqual(session['pages_crawled'], 50)
        self.assertEqual(session['pages_indexed'], 45)
        self.assertEqual(session['pages_skipped'], 5)
        self.assertEqual(session['errors'], 2)
        conn.close()

    def test_list_crawl_sessions(self):
        """list_crawl_sessions should return all sessions."""
        from rag_system.database import (
            get_connection, create_crawl_session, list_crawl_sessions
        )

        conn = get_connection(self.temp_path)
        create_crawl_session(conn, 'https://a.com', ['a.com'], 100)
        create_crawl_session(conn, 'https://b.com', ['b.com'], 200)

        sessions = list_crawl_sessions(conn)

        self.assertEqual(len(sessions), 2)
        conn.close()


class TestCrawlQueueOperations(unittest.TestCase):
    """Test crawl queue operations."""

    def setUp(self):
        """Create a temporary database with a session for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, create_crawl_session
        init_db(self.temp_path)
        conn = get_connection(self.temp_path)
        self.session_id = create_crawl_session(
            conn, 'https://example.com', ['example.com'], 100
        )
        conn.close()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_add_urls_to_crawl_queue(self):
        """add_urls_to_crawl_queue should add URLs to the queue."""
        from rag_system.database import (
            get_connection, add_urls_to_crawl_queue, get_crawl_queue_size
        )

        conn = get_connection(self.temp_path)
        urls = ['https://example.com/a', 'https://example.com/b']
        add_urls_to_crawl_queue(conn, self.session_id, urls)

        size = get_crawl_queue_size(conn, self.session_id)
        self.assertEqual(size, 2)
        conn.close()

    def test_add_urls_to_crawl_queue_with_depth(self):
        """add_urls_to_crawl_queue should store depth."""
        from rag_system.database import (
            get_connection, add_urls_to_crawl_queue, pop_from_crawl_queue
        )

        conn = get_connection(self.temp_path)
        add_urls_to_crawl_queue(
            conn, self.session_id,
            ['https://example.com/deep'],
            depth=3
        )

        url, depth = pop_from_crawl_queue(conn, self.session_id)
        self.assertEqual(depth, 3)
        conn.close()

    def test_add_urls_ignores_duplicates(self):
        """add_urls_to_crawl_queue should ignore duplicate URLs."""
        from rag_system.database import (
            get_connection, add_urls_to_crawl_queue, get_crawl_queue_size
        )

        conn = get_connection(self.temp_path)
        add_urls_to_crawl_queue(conn, self.session_id, ['https://example.com/a'])
        add_urls_to_crawl_queue(conn, self.session_id, ['https://example.com/a'])

        size = get_crawl_queue_size(conn, self.session_id)
        self.assertEqual(size, 1)
        conn.close()

    def test_pop_from_crawl_queue(self):
        """pop_from_crawl_queue should return and remove the oldest URL."""
        from rag_system.database import (
            get_connection, add_urls_to_crawl_queue,
            pop_from_crawl_queue, get_crawl_queue_size
        )

        conn = get_connection(self.temp_path)
        add_urls_to_crawl_queue(
            conn, self.session_id,
            ['https://example.com/first', 'https://example.com/second']
        )

        url, depth = pop_from_crawl_queue(conn, self.session_id)

        self.assertEqual(url, 'https://example.com/first')
        self.assertEqual(get_crawl_queue_size(conn, self.session_id), 1)
        conn.close()

    def test_pop_from_crawl_queue_empty(self):
        """pop_from_crawl_queue should return None when queue is empty."""
        from rag_system.database import get_connection, pop_from_crawl_queue

        conn = get_connection(self.temp_path)
        result = pop_from_crawl_queue(conn, self.session_id)

        self.assertIsNone(result)
        conn.close()

    def test_pop_from_crawl_queue_fifo_order(self):
        """pop_from_crawl_queue should maintain FIFO order."""
        from rag_system.database import (
            get_connection, add_urls_to_crawl_queue, pop_from_crawl_queue
        )

        conn = get_connection(self.temp_path)
        urls = [f'https://example.com/{i}' for i in range(5)]
        add_urls_to_crawl_queue(conn, self.session_id, urls)

        popped = []
        for _ in range(5):
            url, _ = pop_from_crawl_queue(conn, self.session_id)
            popped.append(url)

        self.assertEqual(popped, urls)
        conn.close()

    def test_get_crawl_queue_urls(self):
        """get_crawl_queue_urls should return all URLs in queue."""
        from rag_system.database import (
            get_connection, add_urls_to_crawl_queue, get_crawl_queue_urls
        )

        conn = get_connection(self.temp_path)
        urls = ['https://example.com/a', 'https://example.com/b']
        add_urls_to_crawl_queue(conn, self.session_id, urls)

        queue_urls = get_crawl_queue_urls(conn, self.session_id)

        self.assertEqual(set(queue_urls), set(urls))
        conn.close()

    def test_clear_crawl_queue(self):
        """clear_crawl_queue should remove all URLs from queue."""
        from rag_system.database import (
            get_connection, add_urls_to_crawl_queue,
            clear_crawl_queue, get_crawl_queue_size
        )

        conn = get_connection(self.temp_path)
        add_urls_to_crawl_queue(conn, self.session_id, ['https://example.com/a'])

        clear_crawl_queue(conn, self.session_id)

        size = get_crawl_queue_size(conn, self.session_id)
        self.assertEqual(size, 0)
        conn.close()


class TestCrawlSessionLocking(unittest.TestCase):
    """Test crawl session locking mechanism."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, create_crawl_session
        init_db(self.temp_path)
        conn = get_connection(self.temp_path)
        self.session_id = create_crawl_session(
            conn, 'https://example.com', ['example.com'], 100
        )
        conn.close()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_acquire_session_lock_succeeds_for_interrupted(self):
        """acquire_session_lock should succeed for interrupted session."""
        from rag_system.database import (
            get_connection, update_session_status, acquire_session_lock
        )

        conn = get_connection(self.temp_path)
        update_session_status(conn, self.session_id, 'interrupted')

        success = acquire_session_lock(conn, self.session_id)

        self.assertTrue(success)
        conn.close()

    def test_acquire_session_lock_fails_for_active(self):
        """acquire_session_lock should fail if session is already active."""
        from rag_system.database import get_connection, acquire_session_lock

        conn = get_connection(self.temp_path)

        # Session is already 'active' from creation
        success = acquire_session_lock(conn, self.session_id)

        self.assertFalse(success)
        conn.close()

    def test_release_session_lock(self):
        """release_session_lock should mark session as interrupted."""
        from rag_system.database import (
            get_connection, update_session_status,
            acquire_session_lock, release_session_lock, get_crawl_session
        )

        conn = get_connection(self.temp_path)
        update_session_status(conn, self.session_id, 'interrupted')
        acquire_session_lock(conn, self.session_id)

        release_session_lock(conn, self.session_id)

        session = get_crawl_session(conn, self.session_id)
        self.assertEqual(session['status'], 'interrupted')
        conn.close()


class TestEmbeddingJobsSchema(unittest.TestCase):
    """Test embedding jobs schema creation."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_init_db_creates_embedding_jobs_table(self):
        """init_db should create embedding_jobs table."""
        from rag_system.database import init_db

        init_db(self.temp_path)

        conn = sqlite3.connect(self.temp_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        self.assertIn('embedding_jobs', tables)
        conn.close()

    def test_embedding_jobs_table_has_required_columns(self):
        """embedding_jobs table should have all required columns."""
        from rag_system.database import init_db

        init_db(self.temp_path)

        conn = sqlite3.connect(self.temp_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(embedding_jobs)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            'id', 'page_id', 'status',
            'chunks_total', 'chunks_embedded', 'chunks_skipped', 'chunks_failed',
            'started_at', 'completed_at', 'error_message'
        }
        self.assertTrue(expected_columns.issubset(columns),
                       f"Missing columns: {expected_columns - columns}")
        conn.close()


class TestEmbeddingJobOperations(unittest.TestCase):
    """Test embedding job CRUD operations."""

    def setUp(self):
        """Create a temporary database with a page for testing."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db, get_connection, insert_page
        init_db(self.temp_path)
        conn = get_connection(self.temp_path)
        self.page_id = insert_page(
            conn, 'https://example.com', 'Test Page', '', '', 'abc123'
        )
        conn.close()

    def tearDown(self):
        """Remove temporary database."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_create_embedding_job(self):
        """create_embedding_job should create a new job and return its ID."""
        from rag_system.database import get_connection, create_embedding_job

        conn = get_connection(self.temp_path)
        job_id = create_embedding_job(conn, self.page_id, chunks_total=100)

        self.assertIsInstance(job_id, int)
        self.assertGreater(job_id, 0)
        conn.close()

    def test_create_embedding_job_global(self):
        """create_embedding_job should work with page_id=None for global jobs."""
        from rag_system.database import get_connection, create_embedding_job

        conn = get_connection(self.temp_path)
        job_id = create_embedding_job(conn, None, chunks_total=500)

        self.assertIsInstance(job_id, int)
        self.assertGreater(job_id, 0)
        conn.close()

    def test_get_embedding_job(self):
        """get_embedding_job should retrieve job by ID."""
        from rag_system.database import get_connection, create_embedding_job, get_embedding_job

        conn = get_connection(self.temp_path)
        job_id = create_embedding_job(conn, self.page_id, chunks_total=100)

        job = get_embedding_job(conn, job_id)

        self.assertIsNotNone(job)
        self.assertEqual(job['page_id'], self.page_id)
        self.assertEqual(job['status'], 'in_progress')
        self.assertEqual(job['chunks_total'], 100)
        self.assertEqual(job['chunks_embedded'], 0)
        conn.close()

    def test_get_embedding_job_not_found(self):
        """get_embedding_job should return None for non-existent job."""
        from rag_system.database import get_connection, get_embedding_job

        conn = get_connection(self.temp_path)
        job = get_embedding_job(conn, 9999)

        self.assertIsNone(job)
        conn.close()

    def test_get_active_embedding_job(self):
        """get_active_embedding_job should find in_progress or interrupted jobs."""
        from rag_system.database import (
            get_connection, create_embedding_job, get_active_embedding_job
        )

        conn = get_connection(self.temp_path)
        job_id = create_embedding_job(conn, self.page_id, chunks_total=100)

        active_job = get_active_embedding_job(conn, self.page_id)

        self.assertIsNotNone(active_job)
        self.assertEqual(active_job['id'], job_id)
        conn.close()

    def test_get_active_embedding_job_global(self):
        """get_active_embedding_job should find global jobs when page_id is None."""
        from rag_system.database import (
            get_connection, create_embedding_job, get_active_embedding_job
        )

        conn = get_connection(self.temp_path)
        job_id = create_embedding_job(conn, None, chunks_total=500)

        active_job = get_active_embedding_job(conn, None)

        self.assertIsNotNone(active_job)
        self.assertEqual(active_job['id'], job_id)
        conn.close()

    def test_get_active_embedding_job_ignores_completed(self):
        """get_active_embedding_job should not return completed jobs."""
        from rag_system.database import (
            get_connection, create_embedding_job, get_active_embedding_job,
            update_embedding_job_status
        )

        conn = get_connection(self.temp_path)
        job_id = create_embedding_job(conn, self.page_id, chunks_total=100)
        update_embedding_job_status(conn, job_id, 'completed')

        active_job = get_active_embedding_job(conn, self.page_id)

        self.assertIsNone(active_job)
        conn.close()

    def test_update_embedding_job_progress(self):
        """update_embedding_job_progress should update counters."""
        from rag_system.database import (
            get_connection, create_embedding_job, get_embedding_job,
            update_embedding_job_progress
        )

        conn = get_connection(self.temp_path)
        job_id = create_embedding_job(conn, self.page_id, chunks_total=100)

        update_embedding_job_progress(
            conn, job_id,
            chunks_embedded=50,
            chunks_skipped=10,
            chunks_failed=5
        )

        job = get_embedding_job(conn, job_id)
        self.assertEqual(job['chunks_embedded'], 50)
        self.assertEqual(job['chunks_skipped'], 10)
        self.assertEqual(job['chunks_failed'], 5)
        conn.close()

    def test_update_embedding_job_status(self):
        """update_embedding_job_status should change job status."""
        from rag_system.database import (
            get_connection, create_embedding_job, get_embedding_job,
            update_embedding_job_status
        )

        conn = get_connection(self.temp_path)
        job_id = create_embedding_job(conn, self.page_id, chunks_total=100)

        update_embedding_job_status(conn, job_id, 'completed')

        job = get_embedding_job(conn, job_id)
        self.assertEqual(job['status'], 'completed')
        self.assertIsNotNone(job['completed_at'])
        conn.close()

    def test_update_embedding_job_status_with_error(self):
        """update_embedding_job_status should store error messages."""
        from rag_system.database import (
            get_connection, create_embedding_job, get_embedding_job,
            update_embedding_job_status
        )

        conn = get_connection(self.temp_path)
        job_id = create_embedding_job(conn, self.page_id, chunks_total=100)

        update_embedding_job_status(
            conn, job_id, 'failed',
            error_message='Rate limit exceeded'
        )

        job = get_embedding_job(conn, job_id)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['error_message'], 'Rate limit exceeded')
        conn.close()

    def test_list_embedding_jobs(self):
        """list_embedding_jobs should return all jobs."""
        from rag_system.database import (
            get_connection, create_embedding_job, list_embedding_jobs
        )

        conn = get_connection(self.temp_path)
        create_embedding_job(conn, self.page_id, chunks_total=100)
        create_embedding_job(conn, None, chunks_total=200)

        jobs = list_embedding_jobs(conn)

        self.assertEqual(len(jobs), 2)
        conn.close()

    def test_get_chunks_without_embeddings(self):
        """get_chunks_without_embeddings should return chunks missing embeddings."""
        from rag_system.database import (
            get_connection, insert_chunk, update_chunk_embedding,
            get_chunks_without_embeddings
        )

        conn = get_connection(self.temp_path)

        # Create chunks - one with embedding, one without
        chunk1_id = insert_chunk(
            conn, self.page_id, 'small', 0, 'Content 1', 'Section 1'
        )
        chunk2_id = insert_chunk(
            conn, self.page_id, 'small', 1, 'Content 2', 'Section 2'
        )

        # Add embedding to chunk1 only
        update_chunk_embedding(conn, chunk1_id, [0.1, 0.2, 0.3])

        chunks = get_chunks_without_embeddings(conn)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['id'], chunk2_id)
        conn.close()

    def test_get_chunks_without_embeddings_by_page(self):
        """get_chunks_without_embeddings should filter by page_id."""
        from rag_system.database import (
            get_connection, insert_page, insert_chunk,
            get_chunks_without_embeddings
        )

        conn = get_connection(self.temp_path)

        # Create second page
        page2_id = insert_page(conn, 'https://example.com/page2', 'Page 2', '', '', 'def456')

        # Create chunks on different pages
        insert_chunk(conn, self.page_id, 'small', 0, 'Content 1', 'Section 1')
        insert_chunk(conn, page2_id, 'small', 0, 'Content 2', 'Section 2')

        # Filter by first page only
        chunks = get_chunks_without_embeddings(conn, self.page_id)

        self.assertEqual(len(chunks), 1)
        conn.close()


if __name__ == '__main__':
    unittest.main()
