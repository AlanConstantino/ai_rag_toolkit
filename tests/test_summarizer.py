"""Tests for the summarizer module."""

import unittest
from unittest.mock import MagicMock
import tempfile
import os

from rag_system.database import init_db, get_connection, insert_page


class TestPageSummarizer(unittest.TestCase):
    """Test page summarization."""

    def test_summarize_page_returns_summary(self):
        """summarize_page should return LLM-generated summary."""
        from rag_system.summarization.summarizer import PageSummarizer

        mock_client = MagicMock()
        mock_client.complete.return_value = "Redis is an in-memory cache."

        summarizer = PageSummarizer(chat_client=mock_client)
        summary = summarizer.summarize("Redis is a fast, open-source, in-memory key-value data store...")

        self.assertEqual(summary, "Redis is an in-memory cache.")
        mock_client.complete.assert_called_once()

    def test_summarize_page_truncates_long_text(self):
        """summarize_page should truncate very long text."""
        from rag_system.summarization.summarizer import PageSummarizer

        mock_client = MagicMock()
        mock_client.complete.return_value = "Summary"

        summarizer = PageSummarizer(chat_client=mock_client, max_text_length=100)
        long_text = "x" * 1000
        summarizer.summarize(long_text)

        # Check that the text was truncated
        call_args = mock_client.complete.call_args[0][0]
        self.assertLess(len(call_args), 500)  # Much less than original

    def test_summarize_page_handles_empty_text(self):
        """summarize_page should handle empty text."""
        from rag_system.summarization.summarizer import PageSummarizer

        mock_client = MagicMock()
        summarizer = PageSummarizer(chat_client=mock_client)
        summary = summarizer.summarize("")

        self.assertEqual(summary, "")
        mock_client.complete.assert_not_called()

    def test_summarize_page_handles_error(self):
        """summarize_page should handle API errors gracefully."""
        from rag_system.summarization.summarizer import PageSummarizer

        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("API error")

        summarizer = PageSummarizer(chat_client=mock_client)
        summary = summarizer.summarize("Some content")

        self.assertEqual(summary, "")


class TestSystemSummarizer(unittest.TestCase):
    """Test system summarization."""

    def test_summarize_system_combines_pages(self):
        """summarize_system should combine page summaries."""
        from rag_system.summarization.summarizer import SystemSummarizer

        mock_client = MagicMock()
        mock_client.complete.return_value = "System overview."

        summarizer = SystemSummarizer(chat_client=mock_client)
        page_summaries = [
            {"title": "Installation", "summary": "How to install."},
            {"title": "Configuration", "summary": "How to configure."},
        ]
        summary = summarizer.summarize("Redis", page_summaries)

        self.assertEqual(summary, "System overview.")
        mock_client.complete.assert_called_once()

    def test_summarize_system_handles_empty_pages(self):
        """summarize_system should handle empty page list."""
        from rag_system.summarization.summarizer import SystemSummarizer

        mock_client = MagicMock()
        summarizer = SystemSummarizer(chat_client=mock_client)
        summary = summarizer.summarize("Redis", [])

        self.assertEqual(summary, "")


class TestGlobalSummarizer(unittest.TestCase):
    """Test global summary generation."""

    def test_summarize_global_combines_systems(self):
        """summarize_global should combine system summaries."""
        from rag_system.summarization.summarizer import GlobalSummarizer

        mock_client = MagicMock()
        mock_client.complete.return_value = "Documentation covers Redis and Kafka."

        summarizer = GlobalSummarizer(chat_client=mock_client)
        system_summaries = [
            {"name": "Redis", "summary": "In-memory cache."},
            {"name": "Kafka", "summary": "Message queue."},
        ]
        summary = summarizer.summarize(system_summaries)

        self.assertIn("Redis", summary)
        mock_client.complete.assert_called_once()

    def test_summarize_global_handles_empty_systems(self):
        """summarize_global should handle empty system list."""
        from rag_system.summarization.summarizer import GlobalSummarizer

        mock_client = MagicMock()
        summarizer = GlobalSummarizer(chat_client=mock_client)
        summary = summarizer.summarize([])

        self.assertEqual(summary, "")


class TestSummarizationPipeline(unittest.TestCase):
    """Test full summarization pipeline."""

    def setUp(self):
        """Set up test database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_path)

    def test_summarize_all_pages(self):
        """summarize_all_pages should summarize all pages."""
        from rag_system.summarization.summarizer import SummarizationPipeline

        conn = get_connection(self.temp_path)
        insert_page(conn, url='http://test.com/a', title='Page A',
                   raw_html='', parsed_text='Content A', content_hash='a')
        insert_page(conn, url='http://test.com/b', title='Page B',
                   raw_html='', parsed_text='Content B', content_hash='b')
        conn.close()

        mock_client = MagicMock()
        mock_client.complete.return_value = "Page summary."

        pipeline = SummarizationPipeline(self.temp_path, chat_client=mock_client)
        results = pipeline.summarize_all_pages()

        self.assertEqual(len(results), 2)
        # Should have called complete twice
        self.assertEqual(mock_client.complete.call_count, 2)

    def test_get_page_summaries(self):
        """get_page_summaries should return page summaries."""
        from rag_system.summarization.summarizer import SummarizationPipeline

        conn = get_connection(self.temp_path)
        insert_page(conn, url='http://test.com/a', title='Page A',
                   raw_html='', parsed_text='', content_hash='a',
                   summary='Summary A')
        insert_page(conn, url='http://test.com/b', title='Page B',
                   raw_html='', parsed_text='', content_hash='b',
                   summary='Summary B')
        conn.close()

        pipeline = SummarizationPipeline(self.temp_path)
        summaries = pipeline.get_page_summaries()

        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]['summary'], 'Summary A')


class TestSummaryPrompts(unittest.TestCase):
    """Test summary prompt generation."""

    def test_page_prompt_includes_text(self):
        """Page summary prompt should include the text."""
        from rag_system.summarization.summarizer import build_page_summary_prompt

        prompt = build_page_summary_prompt("Redis is a cache.")
        self.assertIn("Redis is a cache", prompt)

    def test_system_prompt_includes_pages(self):
        """System summary prompt should include page info."""
        from rag_system.summarization.summarizer import build_system_summary_prompt

        pages = [{"title": "Install", "summary": "How to install"}]
        prompt = build_system_summary_prompt("Redis", pages)
        self.assertIn("Redis", prompt)
        self.assertIn("Install", prompt)

    def test_global_prompt_includes_systems(self):
        """Global summary prompt should include systems."""
        from rag_system.summarization.summarizer import build_global_summary_prompt

        systems = [{"name": "Redis", "summary": "Cache system"}]
        prompt = build_global_summary_prompt(systems)
        self.assertIn("Redis", prompt)


if __name__ == '__main__':
    unittest.main()
