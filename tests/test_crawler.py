"""Tests for the crawler module."""

import unittest
from unittest.mock import patch, MagicMock
import urllib.error


class TestURLUtilities(unittest.TestCase):
    """Test URL utility functions."""

    def test_normalize_url_removes_fragment(self):
        """normalize_url should remove URL fragments."""
        from rag_system.ingestion.crawler import normalize_url

        result = normalize_url('https://example.com/page#section')

        self.assertEqual(result, 'https://example.com/page')

    def test_normalize_url_removes_trailing_slash(self):
        """normalize_url should remove trailing slash."""
        from rag_system.ingestion.crawler import normalize_url

        result = normalize_url('https://example.com/page/')

        self.assertEqual(result, 'https://example.com/page')

    def test_normalize_url_preserves_query_params(self):
        """normalize_url should preserve query parameters."""
        from rag_system.ingestion.crawler import normalize_url

        result = normalize_url('https://example.com/page?foo=bar')

        self.assertEqual(result, 'https://example.com/page?foo=bar')

    def test_is_same_domain_true(self):
        """is_same_domain should return True for same domain."""
        from rag_system.ingestion.crawler import is_same_domain

        result = is_same_domain(
            'https://docs.example.com/page',
            ['docs.example.com']
        )

        self.assertTrue(result)

    def test_is_same_domain_false(self):
        """is_same_domain should return False for different domain."""
        from rag_system.ingestion.crawler import is_same_domain

        result = is_same_domain(
            'https://other.com/page',
            ['docs.example.com']
        )

        self.assertFalse(result)

    def test_is_same_domain_subdomain(self):
        """is_same_domain should handle subdomains."""
        from rag_system.ingestion.crawler import is_same_domain

        result = is_same_domain(
            'https://api.docs.example.com/page',
            ['docs.example.com']
        )

        # Subdomain should not match by default
        self.assertFalse(result)

    def test_is_excluded_path(self):
        """is_excluded_path should detect excluded paths."""
        from rag_system.ingestion.crawler import is_excluded_path

        result = is_excluded_path(
            'https://example.com/api/v1/users',
            ['/api/', '/static/']
        )

        self.assertTrue(result)

    def test_is_excluded_path_not_excluded(self):
        """is_excluded_path should return False for non-excluded paths."""
        from rag_system.ingestion.crawler import is_excluded_path

        result = is_excluded_path(
            'https://example.com/docs/guide',
            ['/api/', '/static/']
        )

        self.assertFalse(result)

    def test_extract_links_from_html(self):
        """extract_links should find all href links in HTML."""
        from rag_system.ingestion.crawler import extract_links

        html = '''
        <html>
        <body>
            <a href="/page1">Page 1</a>
            <a href="https://example.com/page2">Page 2</a>
            <a href="#section">Section</a>
        </body>
        </html>
        '''

        links = extract_links(html, 'https://example.com')

        self.assertIn('https://example.com/page1', links)
        self.assertIn('https://example.com/page2', links)
        # Fragment-only links should be filtered out
        self.assertNotIn('https://example.com#section', links)

    def test_extract_links_resolves_relative(self):
        """extract_links should resolve relative URLs."""
        from rag_system.ingestion.crawler import extract_links

        html = '<a href="../other/page">Link</a>'

        links = extract_links(html, 'https://example.com/docs/guide/')

        self.assertIn('https://example.com/docs/other/page', links)


class TestCrawler(unittest.TestCase):
    """Test Crawler class."""

    def test_crawler_initialization(self):
        """Crawler should initialize with config."""
        from rag_system.ingestion.crawler import Crawler

        crawler = Crawler(
            start_url='https://docs.example.com',
            allowed_domains=['docs.example.com'],
            excluded_paths=['/api/'],
            max_pages=100,
            delay=0.5
        )

        self.assertEqual(crawler.start_url, 'https://docs.example.com')
        self.assertEqual(crawler.max_pages, 100)
        self.assertEqual(crawler.delay, 0.5)

    def test_crawler_respects_max_pages(self):
        """Crawler should stop after max_pages."""
        from rag_system.ingestion.crawler import Crawler

        # Mock fetch to return simple HTML with links
        def mock_fetch(url):
            return f'''
            <html>
            <body>
                <a href="/page1">1</a>
                <a href="/page2">2</a>
                <a href="/page3">3</a>
            </body>
            </html>
            ''', 200

        crawler = Crawler(
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=2,
            delay=0
        )

        with patch.object(crawler, '_fetch_url', side_effect=mock_fetch):
            pages = list(crawler.crawl())

        self.assertEqual(len(pages), 2)

    def test_crawler_yields_page_data(self):
        """Crawler should yield page data dictionaries."""
        from rag_system.ingestion.crawler import Crawler

        mock_html = '<html><head><title>Test</title></head><body>Content</body></html>'

        crawler = Crawler(
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=1,
            delay=0
        )

        with patch.object(crawler, '_fetch_url', return_value=(mock_html, 200)):
            pages = list(crawler.crawl())

        self.assertEqual(len(pages), 1)
        self.assertIn('url', pages[0])
        self.assertIn('html', pages[0])
        self.assertIn('status_code', pages[0])

    def test_crawler_skips_non_html(self):
        """Crawler should skip non-HTML content types."""
        from rag_system.ingestion.crawler import Crawler

        crawler = Crawler(
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=5,
            delay=0
        )

        # Mock responses - first HTML, second PDF
        responses = [
            ('<html><a href="/doc.pdf">PDF</a></html>', 200),
            (b'%PDF-1.4', 200),  # PDF content
        ]
        call_count = [0]

        def mock_fetch(url):
            result = responses[min(call_count[0], len(responses) - 1)]
            call_count[0] += 1
            return result

        with patch.object(crawler, '_fetch_url', side_effect=mock_fetch):
            with patch.object(crawler, '_is_html_content', side_effect=[True, False]):
                pages = list(crawler.crawl())

        # Should only get the first HTML page
        self.assertEqual(len(pages), 1)

    def test_crawler_handles_errors(self):
        """Crawler should handle fetch errors gracefully."""
        from rag_system.ingestion.crawler import Crawler

        crawler = Crawler(
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=5,
            delay=0
        )

        # First call succeeds with a link, second call fails
        call_count = [0]

        def mock_fetch(url):
            call_count[0] += 1
            if call_count[0] == 1:
                return '<html><a href="/page2">Link</a></html>', 200
            else:
                raise urllib.error.HTTPError(url, 404, 'Not Found', {}, None)

        with patch.object(crawler, '_fetch_url', side_effect=mock_fetch):
            pages = list(crawler.crawl())

        # Should get first page despite second failing
        self.assertEqual(len(pages), 1)

    def test_crawler_avoids_duplicate_urls(self):
        """Crawler should not visit the same URL twice."""
        from rag_system.ingestion.crawler import Crawler

        # HTML with duplicate links
        html = '''
        <html>
        <body>
            <a href="/page1">Link 1</a>
            <a href="/page1">Link 1 again</a>
            <a href="/page1#section">Link 1 with fragment</a>
        </body>
        </html>
        '''

        crawler = Crawler(
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=10,
            delay=0
        )

        fetch_calls = []

        def mock_fetch(url):
            fetch_calls.append(url)
            return html, 200

        with patch.object(crawler, '_fetch_url', side_effect=mock_fetch):
            list(crawler.crawl())

        # Should only fetch start URL and /page1 once each
        unique_calls = set(fetch_calls)
        self.assertEqual(len(fetch_calls), len(unique_calls))

    def test_crawler_filters_by_domain(self):
        """Crawler should filter out external domains."""
        from rag_system.ingestion.crawler import Crawler

        html = '''
        <html>
        <body>
            <a href="/internal">Internal</a>
            <a href="https://external.com/page">External</a>
        </body>
        </html>
        '''

        crawler = Crawler(
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=10,
            delay=0
        )

        fetch_calls = []

        def mock_fetch(url):
            fetch_calls.append(url)
            return html, 200

        with patch.object(crawler, '_fetch_url', side_effect=mock_fetch):
            list(crawler.crawl())

        # Should not fetch external.com
        self.assertFalse(any('external.com' in url for url in fetch_calls))


class TestRobotsParser(unittest.TestCase):
    """Test robots.txt parsing."""

    def test_can_fetch_allowed(self):
        """can_fetch should return True for allowed paths."""
        from rag_system.ingestion.crawler import RobotsParser

        robots_txt = '''
        User-agent: *
        Allow: /docs/
        Disallow: /admin/
        '''

        parser = RobotsParser('https://example.com', robots_txt)

        self.assertTrue(parser.can_fetch('/docs/guide'))

    def test_can_fetch_disallowed(self):
        """can_fetch should return False for disallowed paths."""
        from rag_system.ingestion.crawler import RobotsParser

        robots_txt = '''
        User-agent: *
        Disallow: /admin/
        '''

        parser = RobotsParser('https://example.com', robots_txt)

        self.assertFalse(parser.can_fetch('/admin/settings'))

    def test_can_fetch_no_robots(self):
        """can_fetch should return True when no robots.txt."""
        from rag_system.ingestion.crawler import RobotsParser

        parser = RobotsParser('https://example.com', None)

        self.assertTrue(parser.can_fetch('/any/path'))

    def test_get_crawl_delay(self):
        """get_crawl_delay should parse delay from robots.txt."""
        from rag_system.ingestion.crawler import RobotsParser

        robots_txt = '''
        User-agent: *
        Crawl-delay: 2
        '''

        parser = RobotsParser('https://example.com', robots_txt)

        self.assertEqual(parser.get_crawl_delay(), 2.0)


if __name__ == '__main__':
    unittest.main()
