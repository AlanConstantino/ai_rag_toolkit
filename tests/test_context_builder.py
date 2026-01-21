"""Tests for the context builder module."""

import unittest
import tempfile
import os

from rag_system.database import (
    init_db, get_connection, insert_page, insert_chunk
)


class TestContextBuilder(unittest.TestCase):
    """Test context building for answer generation."""

    def setUp(self):
        """Set up test database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test Page',
                             raw_html='', parsed_text='Full content',
                             content_hash='a', summary='Page summary')
        self.chunk1_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=0, content='Redis is a cache',
                                      heading_path='Introduction')
        self.chunk2_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=1, content='Configure Redis',
                                      heading_path='Configuration')
        conn.close()

    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_path)

    def test_build_context_from_chunks(self):
        """build_context should combine chunk content."""
        from rag_system.query.context_builder import ContextBuilder

        builder = ContextBuilder(self.temp_path)
        chunks = [
            {'id': self.chunk1_id, 'score': 0.9, 'content': 'Redis is a cache'},
            {'id': self.chunk2_id, 'score': 0.8, 'content': 'Configure Redis'},
        ]
        context = builder.build_context(chunks)

        self.assertIn('Redis is a cache', context)
        self.assertIn('Configure Redis', context)

    def test_build_context_includes_headings(self):
        """build_context should include heading paths."""
        from rag_system.query.context_builder import ContextBuilder

        builder = ContextBuilder(self.temp_path)
        chunks = [{'id': self.chunk1_id, 'score': 0.9}]
        context = builder.build_context(chunks, include_headings=True)

        self.assertIn('Introduction', context)

    def test_build_context_limits_length(self):
        """build_context should respect max_length."""
        from rag_system.query.context_builder import ContextBuilder

        builder = ContextBuilder(self.temp_path, max_context_length=50)
        chunks = [
            {'id': self.chunk1_id, 'score': 0.9, 'content': 'Redis is a cache'},
            {'id': self.chunk2_id, 'score': 0.8, 'content': 'Configure Redis'},
        ]
        context = builder.build_context(chunks)

        self.assertLessEqual(len(context), 100)  # Some margin for formatting


class TestContextFormatting(unittest.TestCase):
    """Test context formatting options."""

    def test_format_chunk_basic(self):
        """format_chunk should format chunk content."""
        from rag_system.query.context_builder import format_chunk

        chunk = {'content': 'Redis is fast', 'heading_path': 'Overview'}
        formatted = format_chunk(chunk)

        self.assertIn('Redis is fast', formatted)

    def test_format_chunk_with_heading(self):
        """format_chunk should include heading when specified."""
        from rag_system.query.context_builder import format_chunk

        chunk = {'content': 'Redis is fast', 'heading_path': 'Overview'}
        formatted = format_chunk(chunk, include_heading=True)

        self.assertIn('Overview', formatted)
        self.assertIn('Redis is fast', formatted)

    def test_format_chunk_with_source(self):
        """format_chunk should include source when specified."""
        from rag_system.query.context_builder import format_chunk

        chunk = {
            'content': 'Redis is fast',
            'page_title': 'Redis Docs',
            'page_url': 'http://test.com'
        }
        formatted = format_chunk(chunk, include_source=True)

        self.assertIn('Redis Docs', formatted)


class TestContextOrdering(unittest.TestCase):
    """Test context chunk ordering."""

    def test_order_by_score(self):
        """order_chunks should sort by score descending."""
        from rag_system.query.context_builder import order_chunks

        chunks = [
            {'id': 1, 'score': 0.5},
            {'id': 2, 'score': 0.9},
            {'id': 3, 'score': 0.7},
        ]
        ordered = order_chunks(chunks, by='score')

        self.assertEqual(ordered[0]['id'], 2)
        self.assertEqual(ordered[1]['id'], 3)
        self.assertEqual(ordered[2]['id'], 1)

    def test_order_by_position(self):
        """order_chunks should sort by chunk_index."""
        from rag_system.query.context_builder import order_chunks

        chunks = [
            {'id': 1, 'chunk_index': 2},
            {'id': 2, 'chunk_index': 0},
            {'id': 3, 'chunk_index': 1},
        ]
        ordered = order_chunks(chunks, by='position')

        self.assertEqual(ordered[0]['id'], 2)
        self.assertEqual(ordered[1]['id'], 3)
        self.assertEqual(ordered[2]['id'], 1)


class TestContextEnrichment(unittest.TestCase):
    """Test context enrichment with metadata."""

    def setUp(self):
        """Set up test database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)
        self.page_id = insert_page(conn, url='http://test.com', title='Test Page',
                                   raw_html='', parsed_text='',
                                   content_hash='a', summary='Summary')
        self.chunk_id = insert_chunk(conn, page_id=self.page_id, chunk_type='small',
                                     chunk_index=0, content='Content',
                                     heading_path='Heading')
        conn.close()

    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_path)

    def test_enrich_chunks_adds_page_info(self):
        """enrich_chunks should add page title and URL."""
        from rag_system.query.context_builder import ContextBuilder

        builder = ContextBuilder(self.temp_path)
        chunks = [{'id': self.chunk_id, 'score': 0.9}]
        enriched = builder.enrich_chunks(chunks)

        self.assertEqual(enriched[0]['page_title'], 'Test Page')
        self.assertEqual(enriched[0]['page_url'], 'http://test.com')


if __name__ == '__main__':
    unittest.main()
