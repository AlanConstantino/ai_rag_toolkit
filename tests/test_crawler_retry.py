"""Tests for crawler retry logic functionality.

Tests the ability to retry failed requests with exponential backoff
for transient errors like 5xx server errors and timeouts.
"""

import unittest
from unittest.mock import patch, MagicMock
import urllib.error

from rag_system.ingestion.crawler import Crawler
from rag_system import config


class TestCrawlerRetryConfig(unittest.TestCase):
    """Tests for crawler retry configuration."""

    def test_max_retries_config_exists(self):
        """CRAWLER_MAX_RETRIES should be configurable."""
        self.assertTrue(hasattr(config, 'CRAWLER_MAX_RETRIES'))
        self.assertIsInstance(config.CRAWLER_MAX_RETRIES, int)
        self.assertGreater(config.CRAWLER_MAX_RETRIES, 0)

    def test_retry_delay_config_exists(self):
        """CRAWLER_RETRY_DELAY should be configurable."""
        self.assertTrue(hasattr(config, 'CRAWLER_RETRY_DELAY'))
        self.assertIsInstance(config.CRAWLER_RETRY_DELAY, (int, float))
        self.assertGreaterEqual(config.CRAWLER_RETRY_DELAY, 0)

    def test_retry_codes_config_exists(self):
        """CRAWLER_RETRY_STATUS_CODES should be configurable."""
        self.assertTrue(hasattr(config, 'CRAWLER_RETRY_STATUS_CODES'))
        self.assertIsInstance(config.CRAWLER_RETRY_STATUS_CODES, (list, tuple, set))


class TestCrawlerRetryLogic(unittest.TestCase):
    """Tests for crawler retry behavior."""

    def setUp(self):
        """Set up crawler for testing."""
        self.crawler = Crawler(
            start_url='https://example.com/',
            allowed_domains=['example.com'],
            max_pages=10,
            delay=0,
            ignore_robots=True
        )

    @patch('rag_system.ingestion.crawler.urllib.request.urlopen')
    @patch('time.sleep')
    def test_retries_on_500_error(self, mock_sleep, mock_urlopen):
        """Crawler should retry on 500 server error."""
        # First call raises 500, second succeeds
        error_response = MagicMock()
        error_response.status = 500
        mock_urlopen.side_effect = [
            urllib.error.HTTPError(
                'https://example.com/', 500, 'Internal Server Error', {}, None
            ),
            MagicMock(
                read=MagicMock(return_value=b'<html><body>Success</body></html>'),
                status=200,
                __enter__=MagicMock(return_value=MagicMock(
                    read=MagicMock(return_value=b'<html><body>Success</body></html>'),
                    status=200
                )),
                __exit__=MagicMock(return_value=False)
            )
        ]

        content, status = self.crawler._fetch_url('https://example.com/')

        # Should have retried
        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(status, 200)

    @patch('rag_system.ingestion.crawler.urllib.request.urlopen')
    @patch('time.sleep')
    def test_retries_on_503_error(self, mock_sleep, mock_urlopen):
        """Crawler should retry on 503 service unavailable."""
        # First call raises 503, second succeeds
        mock_response = MagicMock()
        mock_response.read.return_value = b'<html><body>Success</body></html>'
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            urllib.error.HTTPError(
                'https://example.com/', 503, 'Service Unavailable', {}, None
            ),
            mock_response
        ]

        content, status = self.crawler._fetch_url('https://example.com/')

        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(status, 200)

    @patch('rag_system.ingestion.crawler.urllib.request.urlopen')
    @patch('time.sleep')
    def test_does_not_retry_on_404(self, mock_sleep, mock_urlopen):
        """Crawler should NOT retry on 404 not found."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            'https://example.com/notfound', 404, 'Not Found', {}, None
        )

        with self.assertRaises(urllib.error.HTTPError) as context:
            self.crawler._fetch_url('https://example.com/notfound')

        # Should have only tried once
        self.assertEqual(mock_urlopen.call_count, 1)
        self.assertEqual(context.exception.code, 404)

    @patch('rag_system.ingestion.crawler.urllib.request.urlopen')
    @patch('time.sleep')
    def test_retries_on_timeout(self, mock_sleep, mock_urlopen):
        """Crawler should retry on timeout errors."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'<html><body>Success</body></html>'
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            urllib.error.URLError('timed out'),
            mock_response
        ]

        content, status = self.crawler._fetch_url('https://example.com/')

        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(status, 200)

    @patch('rag_system.ingestion.crawler.urllib.request.urlopen')
    @patch('time.sleep')
    def test_gives_up_after_max_retries(self, mock_sleep, mock_urlopen):
        """Crawler should give up after max retries."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            'https://example.com/', 500, 'Internal Server Error', {}, None
        )

        with patch('rag_system.config.CRAWLER_MAX_RETRIES', 3):
            with self.assertRaises(urllib.error.HTTPError):
                self.crawler._fetch_url('https://example.com/')

        # Should have tried 1 + 3 retries = 4 times total
        self.assertEqual(mock_urlopen.call_count, 4)

    @patch('rag_system.ingestion.crawler.urllib.request.urlopen')
    @patch('time.sleep')
    def test_exponential_backoff(self, mock_sleep, mock_urlopen):
        """Crawler should use exponential backoff between retries."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            'https://example.com/', 500, 'Internal Server Error', {}, None
        )

        with patch('rag_system.config.CRAWLER_MAX_RETRIES', 3):
            with patch('rag_system.config.CRAWLER_RETRY_DELAY', 1.0):
                try:
                    self.crawler._fetch_url('https://example.com/')
                except urllib.error.HTTPError:
                    pass

        # Should have called sleep with exponential delays: 1, 2, 4
        self.assertEqual(mock_sleep.call_count, 3)
        calls = [call[0][0] for call in mock_sleep.call_args_list]
        self.assertEqual(calls, [1.0, 2.0, 4.0])


class TestCrawlerRetryInCrawlMethod(unittest.TestCase):
    """Tests for retry behavior within the crawl() generator."""

    @patch('rag_system.ingestion.crawler.Crawler._fetch_url')
    @patch('time.sleep')
    def test_crawl_continues_after_retry_failure(self, mock_sleep, mock_fetch):
        """crawl() should continue to next URL after exhausting retries."""
        crawler = Crawler(
            start_url='https://example.com/',
            allowed_domains=['example.com'],
            max_pages=5,
            delay=0,
            ignore_robots=True
        )
        crawler.queue = ['https://example.com/page1', 'https://example.com/page2']
        crawler.visited = set()

        # First URL fails completely, second succeeds
        mock_fetch.side_effect = [
            urllib.error.HTTPError(
                'https://example.com/page1', 500, 'Internal Server Error', {}, None
            ),
            ('<html><a href="/page3">link</a></html>', 200)
        ]

        results = list(crawler.crawl())

        # Should have yielded 1 result (page2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['url'], 'https://example.com/page2')


if __name__ == '__main__':
    unittest.main()
