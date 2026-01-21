"""Tests for the main CLI module."""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os


class TestRAGSystem(unittest.TestCase):
    """Test RAGSystem main class."""

    def setUp(self):
        """Set up test database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)

    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_path)

    def test_init_creates_database(self):
        """RAGSystem should initialize database."""
        from rag_system.main import RAGSystem

        rag = RAGSystem(db_path=self.temp_path)

        # Database should exist and have tables
        from rag_system.database import get_connection
        conn = get_connection(self.temp_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row['name'] for row in cursor]
        conn.close()

        self.assertIn('pages', tables)
        self.assertIn('chunks', tables)

    def test_query_returns_result(self):
        """query should return a result dict."""
        from rag_system.main import RAGSystem
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk

        init_db(self.temp_path)
        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='Redis is a cache',
                             content_hash='a')
        insert_chunk(conn, page_id=page_id, chunk_type='small',
                    chunk_index=0, content='Redis is an in-memory cache',
                    heading_path='')
        conn.close()

        rag = RAGSystem(db_path=self.temp_path)
        result = rag.query("What is Redis?")

        self.assertIn('answer', result)
        self.assertIn('chunks', result)
        self.assertIn('confidence', result)

    def test_get_stats(self):
        """get_stats should return system statistics."""
        from rag_system.main import RAGSystem
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk

        init_db(self.temp_path)
        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='Content',
                             content_hash='a')
        insert_chunk(conn, page_id=page_id, chunk_type='small',
                    chunk_index=0, content='Chunk content',
                    heading_path='')
        conn.close()

        rag = RAGSystem(db_path=self.temp_path)
        stats = rag.get_stats()

        self.assertIn('pages', stats)
        self.assertIn('chunks', stats)
        self.assertEqual(stats['pages'], 1)
        self.assertEqual(stats['chunks'], 1)


class TestCLICommands(unittest.TestCase):
    """Test CLI command handling."""

    def test_parse_ingest_command(self):
        """parse_command should handle ingest command."""
        from rag_system.main import parse_command

        cmd, args = parse_command('ingest http://example.com')
        self.assertEqual(cmd, 'ingest')
        self.assertEqual(args, ['http://example.com'])

    def test_parse_query_command(self):
        """parse_command should handle query command."""
        from rag_system.main import parse_command

        cmd, args = parse_command('query What is Redis?')
        self.assertEqual(cmd, 'query')
        self.assertEqual(args, ['What is Redis?'])

    def test_parse_stats_command(self):
        """parse_command should handle stats command."""
        from rag_system.main import parse_command

        cmd, args = parse_command('stats')
        self.assertEqual(cmd, 'stats')
        self.assertEqual(args, [])


class TestArgumentParsing(unittest.TestCase):
    """Test command line argument parsing."""

    def test_create_parser(self):
        """create_parser should create argument parser."""
        from rag_system.main import create_parser

        parser = create_parser()
        self.assertIsNotNone(parser)

    def test_parse_ingest_args(self):
        """Parser should handle ingest arguments."""
        from rag_system.main import create_parser

        parser = create_parser()
        args = parser.parse_args(['ingest', 'http://example.com'])

        self.assertEqual(args.command, 'ingest')
        self.assertEqual(args.url, 'http://example.com')

    def test_parse_query_args(self):
        """Parser should handle query arguments."""
        from rag_system.main import create_parser

        parser = create_parser()
        args = parser.parse_args(['query', 'What is Redis?'])

        self.assertEqual(args.command, 'query')
        self.assertEqual(args.question, 'What is Redis?')


class TestResultFormatting(unittest.TestCase):
    """Test result formatting for display."""

    def test_format_query_result(self):
        """format_query_result should format result for display."""
        from rag_system.main import format_query_result

        result = {
            'answer': 'Redis is a cache.',
            'confidence': 0.8,
            'chunks': [{'content': 'Redis is a cache', 'score': 0.9}]
        }
        formatted = format_query_result(result)

        self.assertIn('Redis is a cache', formatted)
        self.assertIn('confidence', formatted.lower())

    def test_format_stats(self):
        """format_stats should format statistics for display."""
        from rag_system.main import format_stats

        stats = {
            'pages': 10,
            'chunks': 100,
            'entities': 50
        }
        formatted = format_stats(stats)

        self.assertIn('10', formatted)
        self.assertIn('100', formatted)
        self.assertIn('50', formatted)


if __name__ == '__main__':
    unittest.main()
