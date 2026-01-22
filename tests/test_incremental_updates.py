"""Tests for incremental page update functionality.

Tests the ability to detect content changes and re-index pages
without duplicating data.
"""

import unittest
import os
import tempfile

from rag_system.database import (
    init_db, get_connection, insert_page, get_page_by_url,
    insert_chunk, get_chunks_by_page, insert_doc_terms, get_doc_terms,
    delete_chunks_by_page, delete_doc_terms_by_page, update_page_content
)


class TestDeleteChunksByPage(unittest.TestCase):
    """Tests for delete_chunks_by_page function."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.db_fd, self.db_path = tempfile.mkstemp()
        init_db(self.db_path)
        self.conn = get_connection(self.db_path)

    def tearDown(self):
        """Clean up the test database."""
        self.conn.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_delete_all_chunks_for_page(self):
        """delete_chunks_by_page should remove all chunks for a page."""
        # Insert a page
        page_id = insert_page(
            self.conn, url='https://example.com/test',
            title='Test', raw_html='<html></html>',
            parsed_text='Test content', content_hash='abc123'
        )

        # Insert chunks
        insert_chunk(self.conn, page_id, 'large', 0, 'Large chunk', 'Section')
        insert_chunk(self.conn, page_id, 'small', 0, 'Small chunk', 'Section')
        insert_chunk(self.conn, page_id, 'small', 1, 'Another small', 'Section')

        # Verify chunks exist
        chunks = get_chunks_by_page(self.conn, page_id)
        self.assertEqual(len(chunks), 3)

        # Delete chunks
        deleted_count = delete_chunks_by_page(self.conn, page_id)
        self.assertEqual(deleted_count, 3)

        # Verify chunks are gone
        chunks = get_chunks_by_page(self.conn, page_id)
        self.assertEqual(len(chunks), 0)

    def test_delete_only_affects_target_page(self):
        """delete_chunks_by_page should not affect other pages."""
        # Insert two pages
        page1_id = insert_page(
            self.conn, url='https://example.com/page1',
            title='Page 1', raw_html='<html></html>',
            parsed_text='Content 1', content_hash='hash1'
        )
        page2_id = insert_page(
            self.conn, url='https://example.com/page2',
            title='Page 2', raw_html='<html></html>',
            parsed_text='Content 2', content_hash='hash2'
        )

        # Insert chunks for both pages
        insert_chunk(self.conn, page1_id, 'large', 0, 'Page 1 chunk', '')
        insert_chunk(self.conn, page2_id, 'large', 0, 'Page 2 chunk', '')

        # Delete chunks for page 1 only
        delete_chunks_by_page(self.conn, page1_id)

        # Page 1 chunks should be gone
        self.assertEqual(len(get_chunks_by_page(self.conn, page1_id)), 0)
        # Page 2 chunks should remain
        self.assertEqual(len(get_chunks_by_page(self.conn, page2_id)), 1)

    def test_delete_returns_zero_for_empty_page(self):
        """delete_chunks_by_page should return 0 if page has no chunks."""
        page_id = insert_page(
            self.conn, url='https://example.com/empty',
            title='Empty', raw_html='<html></html>',
            parsed_text='', content_hash='empty'
        )

        deleted_count = delete_chunks_by_page(self.conn, page_id)
        self.assertEqual(deleted_count, 0)


class TestDeleteDocTermsByPage(unittest.TestCase):
    """Tests for delete_doc_terms_by_page function."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.db_fd, self.db_path = tempfile.mkstemp()
        init_db(self.db_path)
        self.conn = get_connection(self.db_path)

    def tearDown(self):
        """Clean up the test database."""
        self.conn.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_delete_doc_terms_for_page(self):
        """delete_doc_terms_by_page should remove BM25 terms for page chunks."""
        page_id = insert_page(
            self.conn, url='https://example.com/test',
            title='Test', raw_html='<html></html>',
            parsed_text='Test content', content_hash='abc123'
        )

        # Insert chunks and their BM25 terms
        chunk1_id = insert_chunk(self.conn, page_id, 'small', 0, 'Hello world', '')
        chunk2_id = insert_chunk(self.conn, page_id, 'small', 1, 'Foo bar', '')

        insert_doc_terms(self.conn, chunk1_id, {'hello': 1, 'world': 1})
        insert_doc_terms(self.conn, chunk2_id, {'foo': 1, 'bar': 1})

        # Verify terms exist
        self.assertEqual(get_doc_terms(self.conn, chunk1_id), {'hello': 1, 'world': 1})
        self.assertEqual(get_doc_terms(self.conn, chunk2_id), {'foo': 1, 'bar': 1})

        # Delete doc terms for page
        deleted_count = delete_doc_terms_by_page(self.conn, page_id)
        self.assertEqual(deleted_count, 4)  # 4 total terms

        # Verify terms are gone
        self.assertEqual(get_doc_terms(self.conn, chunk1_id), {})
        self.assertEqual(get_doc_terms(self.conn, chunk2_id), {})


class TestUpdatePageContent(unittest.TestCase):
    """Tests for update_page_content function."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.db_fd, self.db_path = tempfile.mkstemp()
        init_db(self.db_path)
        self.conn = get_connection(self.db_path)

    def tearDown(self):
        """Clean up the test database."""
        self.conn.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_update_page_content(self):
        """update_page_content should update page fields."""
        page_id = insert_page(
            self.conn, url='https://example.com/test',
            title='Old Title', raw_html='<html>old</html>',
            parsed_text='Old content', content_hash='old_hash'
        )

        update_page_content(
            self.conn, page_id,
            title='New Title',
            raw_html='<html>new</html>',
            parsed_text='New content',
            content_hash='new_hash'
        )

        page = get_page_by_url(self.conn, 'https://example.com/test')
        self.assertEqual(page['title'], 'New Title')
        self.assertEqual(page['raw_html'], '<html>new</html>')
        self.assertEqual(page['parsed_text'], 'New content')
        self.assertEqual(page['content_hash'], 'new_hash')


class TestIncrementalUpdateIntegration(unittest.TestCase):
    """Integration tests for incremental page updates."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.db_fd, self.db_path = tempfile.mkstemp()
        init_db(self.db_path)
        self.conn = get_connection(self.db_path)

    def tearDown(self):
        """Clean up the test database."""
        self.conn.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_full_update_workflow(self):
        """Full workflow: delete old data, update page, insert new data."""
        # Initial state: page with chunks and BM25 terms
        page_id = insert_page(
            self.conn, url='https://example.com/test',
            title='Version 1', raw_html='<html>v1</html>',
            parsed_text='Version 1 content', content_hash='v1_hash'
        )

        chunk_id = insert_chunk(self.conn, page_id, 'small', 0, 'Old chunk', 'Old Section')
        insert_doc_terms(self.conn, chunk_id, {'old': 1, 'chunk': 1})

        # Simulate content change detection and update
        # Step 1: Delete old BM25 terms
        delete_doc_terms_by_page(self.conn, page_id)

        # Step 2: Delete old chunks
        delete_chunks_by_page(self.conn, page_id)

        # Step 3: Update page content
        update_page_content(
            self.conn, page_id,
            title='Version 2',
            raw_html='<html>v2</html>',
            parsed_text='Version 2 content',
            content_hash='v2_hash'
        )

        # Step 4: Insert new chunks
        new_chunk_id = insert_chunk(
            self.conn, page_id, 'small', 0, 'New chunk', 'New Section'
        )
        insert_doc_terms(self.conn, new_chunk_id, {'new': 1, 'chunk': 1})

        # Verify final state
        page = get_page_by_url(self.conn, 'https://example.com/test')
        self.assertEqual(page['title'], 'Version 2')
        self.assertEqual(page['content_hash'], 'v2_hash')

        chunks = get_chunks_by_page(self.conn, page_id)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['content'], 'New chunk')

        terms = get_doc_terms(self.conn, new_chunk_id)
        self.assertEqual(terms, {'new': 1, 'chunk': 1})


class TestIndexerIncrementalUpdate(unittest.TestCase):
    """Tests for Indexer incremental update functionality."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.db_fd, self.db_path = tempfile.mkstemp()
        init_db(self.db_path)

    def tearDown(self):
        """Clean up the test database."""
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_indexer_skips_unchanged_page(self):
        """Indexer should skip pages with unchanged content."""
        from rag_system.ingestion.indexer import Indexer

        indexer = Indexer(self.db_path)

        # Index a page
        page_data = {
            'url': 'https://example.com/test',
            'html': '<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>',
            'status_code': 200
        }
        page_id = indexer.index_page(page_data)
        self.assertIsNotNone(page_id)

        # Try to index the same page again
        result = indexer.index_page(page_data)
        self.assertIsNone(result)  # Should return None (skipped)

    def test_indexer_updates_changed_page(self):
        """Indexer should re-index pages with changed content."""
        from rag_system.ingestion.indexer import Indexer

        indexer = Indexer(self.db_path)

        # Index initial version
        page_data_v1 = {
            'url': 'https://example.com/test',
            'html': '<html><head><title>V1</title></head><body><h1>Version 1</h1></body></html>',
            'status_code': 200
        }
        page_id = indexer.index_page(page_data_v1)
        self.assertIsNotNone(page_id)

        # Get initial chunk count
        conn = get_connection(self.db_path)
        initial_chunks = get_chunks_by_page(conn, page_id)
        initial_count = len(initial_chunks)
        conn.close()

        # Index updated version (different content)
        page_data_v2 = {
            'url': 'https://example.com/test',
            'html': '<html><head><title>V2</title></head><body><h1>Version 2</h1><p>New content</p></body></html>',
            'status_code': 200
        }
        result = indexer.index_page(page_data_v2)

        # Should return None because we're updating, not creating
        # (the page_id stays the same, chunks are recreated)
        # Actually, let's verify the page was updated
        conn = get_connection(self.db_path)
        page = get_page_by_url(conn, 'https://example.com/test')
        self.assertEqual(page['title'], 'V2')
        self.assertIn('Version 2', page['parsed_text'])

        # Should have new chunks (old ones deleted)
        updated_chunks = get_chunks_by_page(conn, page_id)
        self.assertTrue(len(updated_chunks) > 0)
        conn.close()


if __name__ == '__main__':
    unittest.main()
