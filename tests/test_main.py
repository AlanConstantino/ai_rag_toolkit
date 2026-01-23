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

    def test_ingest_with_mocked_crawler(self):
        """ingest should call crawl_and_index with correct parameters."""
        from rag_system.main import RAGSystem
        from rag_system.ingestion.indexer import Indexer

        rag = RAGSystem(db_path=self.temp_path)

        # Mock crawl_and_index to verify it's called correctly
        with patch.object(Indexer, 'crawl_and_index') as mock_crawl:
            mock_crawl.return_value = {
                'pages_crawled': 5,
                'pages_indexed': 4,
                'pages_skipped': 1,
                'errors': 0
            }

            stats = rag.ingest('https://docs.python.org/3.6/', max_pages=5)

            # Verify crawl_and_index was called
            self.assertTrue(mock_crawl.called)

            # Verify parameters passed correctly
            call_args = mock_crawl.call_args
            self.assertEqual(call_args[1]['start_url'], 'https://docs.python.org/3.6/')
            self.assertIn('allowed_domains', call_args[1])
            self.assertEqual(call_args[1]['allowed_domains'], ['docs.python.org'])
            self.assertEqual(call_args[1]['max_pages'], 5)

            # Verify stats returned
            self.assertEqual(stats['pages_crawled'], 5)
            self.assertEqual(stats['pages_indexed'], 4)


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


class TestIngestOutput(unittest.TestCase):
    """Test ingestion output formatting."""

    def test_stats_printed_correctly(self):
        """main() should print correct stats after ingestion."""
        from rag_system.main import main
        import sys
        from io import StringIO

        # Mock sys.argv for CLI
        test_args = ['rag_system', 'ingest', 'https://example.com', '--max-pages', '5']

        # Mock RAGSystem.ingest to return stats
        mock_stats = {
            'pages_crawled': 10,
            'pages_indexed': 8,
            'pages_skipped': 2,
            'errors': 0
        }

        with patch('sys.argv', test_args), \
             patch('rag_system.main.RAGSystem.ingest', return_value=mock_stats), \
             patch('sys.stdout', new=StringIO()) as fake_out:

            main()
            output = fake_out.getvalue()

            # Verify output contains correct numbers
            self.assertIn('Crawled 10 pages', output)  # pages_crawled
            self.assertIn('indexed 8', output)   # pages_indexed
            self.assertNotIn('Ingested 0 pages', output)  # Bug 2 check - old buggy format


class TestCrawlSessionCLI(unittest.TestCase):
    """Test crawl session CLI commands."""

    def setUp(self):
        """Set up test database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        from rag_system.database import init_db
        init_db(self.temp_path)

    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_path)

    def test_ingest_fresh_flag_parsed(self):
        """Parser should recognize --fresh flag."""
        from rag_system.main import create_parser

        parser = create_parser()
        args = parser.parse_args(['ingest', 'https://example.com', '--fresh'])

        self.assertTrue(args.fresh)

    def test_ingest_fresh_flag_default_false(self):
        """Parser should default --fresh to False."""
        from rag_system.main import create_parser

        parser = create_parser()
        args = parser.parse_args(['ingest', 'https://example.com'])

        self.assertFalse(args.fresh)

    def test_ingest_passes_fresh_to_indexer(self):
        """ingest command should pass fresh flag to indexer."""
        from rag_system.main import RAGSystem
        from rag_system.ingestion.indexer import Indexer

        rag = RAGSystem(db_path=self.temp_path)

        with patch.object(Indexer, 'crawl_and_index') as mock_crawl:
            mock_crawl.return_value = {
                'pages_crawled': 1,
                'pages_indexed': 1,
                'pages_skipped': 0,
                'errors': 0
            }

            rag.ingest('https://example.com', max_pages=10, fresh=True)

            mock_crawl.assert_called_once()
            call_kwargs = mock_crawl.call_args[1]
            self.assertTrue(call_kwargs.get('fresh', False))

    def test_crawl_sessions_command_exists(self):
        """Parser should have crawl-sessions command."""
        from rag_system.main import create_parser

        parser = create_parser()
        args = parser.parse_args(['crawl-sessions'])

        self.assertEqual(args.command, 'crawl-sessions')

    def test_crawl_sessions_list(self):
        """crawl-sessions should list sessions."""
        from rag_system.main import main
        from rag_system.database import get_connection, create_crawl_session
        from io import StringIO

        # Create a session
        conn = get_connection(self.temp_path)
        create_crawl_session(conn, 'https://example.com', ['example.com'], 100)
        conn.close()

        test_args = ['rag_system', '--db', self.temp_path, 'crawl-sessions']

        with patch('sys.argv', test_args), \
             patch('sys.stdout', new=StringIO()) as fake_out:
            main()
            output = fake_out.getvalue()

            self.assertIn('https://example.com', output)

    def test_crawl_sessions_json_output(self):
        """crawl-sessions --json should output JSON."""
        from rag_system.main import main
        from rag_system.database import get_connection, create_crawl_session
        from io import StringIO
        import json

        conn = get_connection(self.temp_path)
        create_crawl_session(conn, 'https://example.com', ['example.com'], 100)
        conn.close()

        test_args = ['rag_system', '--db', self.temp_path, 'crawl-sessions', '--json']

        with patch('sys.argv', test_args), \
             patch('sys.stdout', new=StringIO()) as fake_out:
            main()
            output = fake_out.getvalue()

            # Should be valid JSON
            data = json.loads(output)
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]['start_url'], 'https://example.com')


if __name__ == '__main__':
    unittest.main()
