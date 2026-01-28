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
        from rag_system.ingestion.crawler import Crawler, FetchError

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
                # Raise FetchError since mock bypasses the conversion logic
                raise FetchError(url, 'HTTP 404: Not Found', status_code=404)

        with patch.object(crawler, '_fetch_url', side_effect=mock_fetch):
            pages = list(crawler.crawl())

        # Should get first page despite second failing
        self.assertEqual(len(pages), 1)

        # Check that stats recorded the failure
        stats = crawler.get_stats()
        self.assertEqual(stats.pages_succeeded, 1)
        self.assertEqual(stats.pages_failed, 1)
        self.assertEqual(len(stats.errors), 1)

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


class TestCrawlerExceptions(unittest.TestCase):
    """Test custom crawler exceptions."""

    def test_crawler_error_is_base_exception(self):
        """CrawlerError should be the base exception."""
        from rag_system.ingestion.crawler import CrawlerError, FetchError, CacheError

        self.assertTrue(issubclass(FetchError, CrawlerError))
        self.assertTrue(issubclass(CacheError, CrawlerError))

    def test_fetch_error_has_attributes(self):
        """FetchError should store url and status_code."""
        from rag_system.ingestion.crawler import FetchError

        error = FetchError(url='https://example.com', message='Not Found', status_code=404)

        self.assertEqual(error.url, 'https://example.com')
        self.assertEqual(error.status_code, 404)
        self.assertIn('https://example.com', str(error))

    def test_cache_error_has_url(self):
        """CacheError should store url."""
        from rag_system.ingestion.crawler import CacheError

        error = CacheError(url='https://example.com', message='Corrupted')

        self.assertEqual(error.url, 'https://example.com')

    def test_url_validation_error_has_reason(self):
        """URLValidationError should store url and reason."""
        from rag_system.ingestion.crawler import URLValidationError

        error = URLValidationError(url='file:///etc/passwd', reason='Invalid scheme')

        self.assertEqual(error.url, 'file:///etc/passwd')
        self.assertEqual(error.reason, 'Invalid scheme')


class TestURLValidation(unittest.TestCase):
    """Test URL validation functions."""

    def test_validate_url_valid_https(self):
        """validate_url should accept valid https URLs."""
        from rag_system.ingestion.crawler import validate_url

        is_valid, error = validate_url('https://example.com/page')

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_validate_url_valid_http(self):
        """validate_url should accept valid http URLs."""
        from rag_system.ingestion.crawler import validate_url

        is_valid, error = validate_url('http://example.com/page')

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_validate_url_rejects_file_scheme(self):
        """validate_url should reject file:// URLs."""
        from rag_system.ingestion.crawler import validate_url

        is_valid, error = validate_url('file:///etc/passwd')

        self.assertFalse(is_valid)
        self.assertIn('scheme', error.lower())

    def test_validate_url_rejects_ftp_scheme(self):
        """validate_url should reject ftp:// URLs."""
        from rag_system.ingestion.crawler import validate_url

        is_valid, error = validate_url('ftp://example.com/file')

        self.assertFalse(is_valid)
        self.assertIn('scheme', error.lower())

    def test_validate_url_rejects_localhost(self):
        """validate_url should reject localhost by default."""
        from rag_system.ingestion.crawler import validate_url

        is_valid, error = validate_url('http://localhost/admin')

        self.assertFalse(is_valid)
        self.assertIn('blocked', error.lower())

    def test_validate_url_rejects_metadata_hostname(self):
        """validate_url should reject cloud metadata hostnames."""
        from rag_system.ingestion.crawler import validate_url

        is_valid, error = validate_url('http://metadata.google.internal/')

        self.assertFalse(is_valid)
        self.assertIn('blocked', error.lower())

    def test_validate_url_allows_private_when_enabled(self):
        """validate_url should allow localhost when allow_private=True."""
        from rag_system.ingestion.crawler import validate_url

        is_valid, error = validate_url('http://localhost/admin', allow_private=True)

        self.assertTrue(is_valid)

    def test_validate_url_or_raise_raises(self):
        """validate_url_or_raise should raise URLValidationError."""
        from rag_system.ingestion.crawler import validate_url_or_raise, URLValidationError

        with self.assertRaises(URLValidationError) as ctx:
            validate_url_or_raise('file:///etc/passwd')

        self.assertEqual(ctx.exception.url, 'file:///etc/passwd')

    def test_validate_url_rejects_spaces(self):
        """validate_url should reject URLs containing spaces."""
        from rag_system.ingestion.crawler import validate_url

        # URL with space (like the malformed Python docs link)
        is_valid, error = validate_url('https://example.com/path/ https:/other.com/path')

        self.assertFalse(is_valid)
        self.assertIn('space', error.lower())

    def test_validate_url_rejects_control_characters(self):
        """validate_url should reject URLs containing control characters."""
        from rag_system.ingestion.crawler import validate_url

        # URL with tab character
        is_valid, error = validate_url('https://example.com/path\twith\ttabs')

        self.assertFalse(is_valid)
        self.assertIn('control character', error.lower())

    def test_validate_url_rejects_newline(self):
        """validate_url should reject URLs containing newlines."""
        from rag_system.ingestion.crawler import validate_url

        # URL with newline
        is_valid, error = validate_url('https://example.com/path\nwith\nnewlines')

        self.assertFalse(is_valid)
        self.assertIn('control character', error.lower())


class TestCrawlStats(unittest.TestCase):
    """Test CrawlStats class."""

    def test_crawl_stats_initialization(self):
        """CrawlStats should initialize with zeros."""
        from rag_system.ingestion.crawler import CrawlStats

        stats = CrawlStats()

        self.assertEqual(stats.pages_succeeded, 0)
        self.assertEqual(stats.pages_failed, 0)
        self.assertEqual(stats.pages_skipped, 0)
        self.assertEqual(len(stats.errors), 0)

    def test_crawl_stats_record_success(self):
        """record_success should increment pages_succeeded."""
        from rag_system.ingestion.crawler import CrawlStats

        stats = CrawlStats()
        stats.record_success()
        stats.record_success()

        self.assertEqual(stats.pages_succeeded, 2)

    def test_crawl_stats_record_failure(self):
        """record_failure should increment pages_failed and add error."""
        from rag_system.ingestion.crawler import CrawlStats

        stats = CrawlStats()
        stats.record_failure('https://example.com/page', 'HTTP 404')

        self.assertEqual(stats.pages_failed, 1)
        self.assertEqual(len(stats.errors), 1)
        self.assertEqual(stats.errors[0]['url'], 'https://example.com/page')
        self.assertEqual(stats.errors[0]['error'], 'HTTP 404')

    def test_crawl_stats_record_skip(self):
        """record_skip should increment pages_skipped."""
        from rag_system.ingestion.crawler import CrawlStats

        stats = CrawlStats()
        stats.record_skip()

        self.assertEqual(stats.pages_skipped, 1)

    def test_crawl_stats_to_dict(self):
        """to_dict should return all stats as dictionary."""
        from rag_system.ingestion.crawler import CrawlStats

        stats = CrawlStats()
        stats.record_success()
        stats.record_failure('https://example.com', 'error')
        stats.record_skip()

        result = stats.to_dict()

        self.assertEqual(result['pages_succeeded'], 1)
        self.assertEqual(result['pages_failed'], 1)
        self.assertEqual(result['pages_skipped'], 1)
        self.assertEqual(result['total_errors'], 1)
        self.assertEqual(len(result['errors']), 1)


class TestHTTPCache(unittest.TestCase):
    """Test HTTPCache class."""

    def setUp(self):
        """Set up test cache directory."""
        import tempfile
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test cache directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_put_and_get(self):
        """HTTPCache should store and retrieve responses."""
        from rag_system.ingestion.crawler import HTTPCache

        cache = HTTPCache(self.temp_dir)
        cache.put('https://example.com/page', '<html>content</html>', 200)

        result = cache.get('https://example.com/page')

        self.assertIsNotNone(result)
        self.assertEqual(result[0], '<html>content</html>')
        self.assertEqual(result[1], 200)

    def test_cache_get_missing(self):
        """HTTPCache should return None for missing URLs."""
        from rag_system.ingestion.crawler import HTTPCache

        cache = HTTPCache(self.temp_dir)

        result = cache.get('https://example.com/nonexistent')

        self.assertIsNone(result)

    def test_cache_handles_corrupted_json(self):
        """HTTPCache should handle corrupted cache files."""
        from rag_system.ingestion.crawler import HTTPCache
        import hashlib
        import os

        cache = HTTPCache(self.temp_dir)
        url = 'https://example.com/corrupted'
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = os.path.join(self.temp_dir, f"{url_hash}.json")

        # Write corrupted JSON
        with open(cache_path, 'w') as f:
            f.write('not valid json {{{')

        # Should return None and remove corrupted file
        result = cache.get(url)

        self.assertIsNone(result)
        # Corrupted file should be removed
        self.assertFalse(os.path.exists(cache_path))

    def test_cache_handles_empty_file(self):
        """HTTPCache should handle empty cache files."""
        from rag_system.ingestion.crawler import HTTPCache
        import hashlib
        import os

        cache = HTTPCache(self.temp_dir)
        url = 'https://example.com/empty'
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = os.path.join(self.temp_dir, f"{url_hash}.json")

        # Write empty file
        with open(cache_path, 'w') as f:
            f.write('')

        # Should return None
        result = cache.get(url)

        self.assertIsNone(result)

    def test_cache_handles_missing_fields(self):
        """HTTPCache should handle cache files with missing fields."""
        from rag_system.ingestion.crawler import HTTPCache
        import hashlib
        import os
        import json

        cache = HTTPCache(self.temp_dir)
        url = 'https://example.com/incomplete'
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = os.path.join(self.temp_dir, f"{url_hash}.json")

        # Write JSON with missing required fields
        with open(cache_path, 'w') as f:
            json.dump({'url': url}, f)  # Missing 'content' and 'status'

        # Should return None
        result = cache.get(url)

        self.assertIsNone(result)


class TestCrawlerWithStats(unittest.TestCase):
    """Test Crawler class with statistics tracking."""

    def test_crawler_tracks_success_stats(self):
        """Crawler should track successful page fetches."""
        from rag_system.ingestion.crawler import Crawler

        crawler = Crawler(
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=2,
            delay=0
        )

        def mock_fetch(url):
            return '<html><a href="/page2">Link</a></html>', 200

        with patch.object(crawler, '_fetch_url', side_effect=mock_fetch):
            list(crawler.crawl())

        stats = crawler.get_stats()
        self.assertEqual(stats.pages_succeeded, 2)
        self.assertEqual(stats.pages_failed, 0)

    def test_crawler_tracks_failure_stats(self):
        """Crawler should track failed page fetches."""
        from rag_system.ingestion.crawler import Crawler, FetchError

        crawler = Crawler(
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=5,
            delay=0
        )

        call_count = [0]

        def mock_fetch(url):
            call_count[0] += 1
            if call_count[0] <= 1:
                return '<html><a href="/page2">Link</a><a href="/page3">Link</a></html>', 200
            else:
                raise FetchError(url, 'HTTP 500', status_code=500)

        with patch.object(crawler, '_fetch_url', side_effect=mock_fetch):
            list(crawler.crawl())

        stats = crawler.get_stats()
        self.assertEqual(stats.pages_succeeded, 1)
        self.assertEqual(stats.pages_failed, 2)

    def test_get_stats_returns_crawl_stats(self):
        """get_stats should return CrawlStats object."""
        from rag_system.ingestion.crawler import Crawler, CrawlStats

        crawler = Crawler(
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=1,
            delay=0
        )

        stats = crawler.get_stats()

        self.assertIsInstance(stats, CrawlStats)


class TestCrawlerURLValidation(unittest.TestCase):
    """Test Crawler URL validation integration."""

    def test_crawler_validates_start_url(self):
        """Crawler should validate URLs before fetching."""
        from rag_system.ingestion.crawler import Crawler, URLValidationError

        crawler = Crawler(
            start_url='https://example.com',
            allowed_domains=['example.com'],
            max_pages=5,
            delay=0
        )

        # Try to fetch a URL that should fail validation
        with self.assertRaises(URLValidationError):
            crawler._fetch_url('file:///etc/passwd')

    def test_crawler_allows_private_when_configured(self):
        """Crawler should allow private URLs when allow_private_urls=True."""
        from rag_system.ingestion.crawler import Crawler

        crawler = Crawler(
            start_url='http://localhost',
            allowed_domains=['localhost'],
            max_pages=1,
            delay=0,
            allow_private_urls=True
        )

        # Mock the network fetch to avoid actual connection
        def mock_network_fetch(url):
            return '<html>test</html>', 200

        with patch.object(crawler, '_fetch_url_from_network', side_effect=mock_network_fetch):
            # Should not raise URLValidationError
            content, status = crawler._fetch_url('http://localhost/page')

        self.assertEqual(status, 200)


if __name__ == '__main__':
    unittest.main()
