"""Web crawler module for the RAG system.

Provides BFS web crawling with robots.txt respect, rate limiting,
and domain filtering. Uses only Python standard library.
"""

import functools
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Dict, Generator, List, Optional, Set, Tuple
import hashlib
import os
import json
from pathlib import Path

from rag_system.utils import get_logger
from rag_system import config
from rag_system.shutdown import is_shutdown_requested

logger = get_logger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class CrawlerError(Exception):
    """Base exception for crawler operations."""
    pass


class FetchError(CrawlerError):
    """Exception raised when fetching a URL fails."""

    def __init__(self, url: str, message: str, status_code: Optional[int] = None):
        self.url = url
        self.status_code = status_code
        super().__init__(f"Failed to fetch {url}: {message}")


class CacheError(CrawlerError):
    """Exception raised for cache-related errors."""

    def __init__(self, url: str, message: str):
        self.url = url
        super().__init__(f"Cache error for {url}: {message}")


class URLValidationError(CrawlerError):
    """Exception raised when URL validation fails."""

    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"Invalid URL {url}: {reason}")


# =============================================================================
# Crawl Statistics
# =============================================================================

class CrawlStats:
    """Statistics from a crawl operation."""

    def __init__(self):
        self.pages_succeeded: int = 0
        self.pages_failed: int = 0
        self.pages_skipped: int = 0
        self.errors: List[Dict[str, str]] = []

    def record_success(self) -> None:
        """Record a successful page fetch."""
        self.pages_succeeded += 1

    def record_failure(self, url: str, error: str) -> None:
        """Record a failed page fetch."""
        self.pages_failed += 1
        self.errors.append({'url': url, 'error': error})

    def record_skip(self) -> None:
        """Record a skipped page."""
        self.pages_skipped += 1

    def to_dict(self) -> Dict:
        """Convert stats to dictionary."""
        return {
            'pages_succeeded': self.pages_succeeded,
            'pages_failed': self.pages_failed,
            'pages_skipped': self.pages_skipped,
            'total_errors': len(self.errors),
            'errors': self.errors
        }


# =============================================================================
# Retry Decorator
# =============================================================================

def retry_on_error(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exponential_backoff: bool = True,
    retryable_exceptions: Tuple = (urllib.error.URLError,)
):
    """Decorator to retry operations on transient errors.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.
        exponential_backoff: If True, use exponential backoff for delays.
        retryable_exceptions: Tuple of exception types to retry on.

    Returns:
        Decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except urllib.error.HTTPError as e:
                    # Only retry on retryable status codes
                    if e.code not in config.CRAWLER_RETRY_STATUS_CODES:
                        raise FetchError(
                            url=getattr(e, 'url', str(args[1] if len(args) > 1 else 'unknown')),
                            message=f"HTTP {e.code}: {e.reason}",
                            status_code=e.code
                        )
                    last_exception = e
                    if attempt >= max_retries:
                        raise FetchError(
                            url=getattr(e, 'url', str(args[1] if len(args) > 1 else 'unknown')),
                            message=f"HTTP {e.code} after {max_retries} retries",
                            status_code=e.code
                        )
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt >= max_retries:
                        url = str(args[1]) if len(args) > 1 else 'unknown'
                        raise FetchError(
                            url=url,
                            message=f"{type(e).__name__}: {e}"
                        )

                # Calculate delay with exponential backoff
                if exponential_backoff:
                    delay = base_delay * (2 ** attempt)
                else:
                    delay = base_delay

                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after error: {last_exception}, "
                    f"waiting {delay}s"
                )
                time.sleep(delay)

            # Should not reach here, but just in case
            raise last_exception

        return wrapper
    return decorator


# =============================================================================
# URL Validation
# =============================================================================

# Private IP ranges that should be blocked to prevent SSRF
PRIVATE_IP_PREFIXES = (
    '10.',
    '172.16.', '172.17.', '172.18.', '172.19.',
    '172.20.', '172.21.', '172.22.', '172.23.',
    '172.24.', '172.25.', '172.26.', '172.27.',
    '172.28.', '172.29.', '172.30.', '172.31.',
    '192.168.',
    '127.',
    '0.',
    '169.254.',  # Link-local
)

# Blocked hostnames that could be used for SSRF
BLOCKED_HOSTNAMES = {
    'localhost',
    'localhost.localdomain',
    'metadata.google.internal',  # GCP metadata
    'metadata',  # Generic cloud metadata
}


def validate_url(url: str, allow_private: bool = False) -> Tuple[bool, Optional[str]]:
    """Validate a URL for safe crawling.

    Checks for:
    - Valid URL scheme (http/https only)
    - Non-private IP addresses (unless allow_private is True)
    - Non-blocked hostnames

    Args:
        url: URL to validate.
        allow_private: If True, allow private/internal addresses.

    Returns:
        Tuple of (is_valid, error_message).
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        return False, f"Failed to parse URL: {e}"

    # Check scheme
    if parsed.scheme not in ('http', 'https'):
        return False, f"Invalid scheme: {parsed.scheme}"

    # Check for empty host
    if not parsed.netloc:
        return False, "Empty host"

    hostname = parsed.netloc.split(':')[0].lower()

    if not allow_private:
        # Check blocked hostnames only when private addresses are not allowed
        if hostname in BLOCKED_HOSTNAMES:
            return False, f"Blocked hostname: {hostname}"
        # Check if hostname is an IP address
        try:
            # Try to resolve and check for private IP
            ip_addr = socket.gethostbyname(hostname)
            for prefix in PRIVATE_IP_PREFIXES:
                if ip_addr.startswith(prefix):
                    return False, f"Private IP address: {ip_addr}"
        except socket.gaierror:
            # Could not resolve - hostname doesn't exist
            # This is fine for validation, let the fetch fail naturally
            pass
        except socket.herror:
            pass

    return True, None


def validate_url_or_raise(url: str, allow_private: bool = False) -> None:
    """Validate a URL and raise URLValidationError if invalid.

    Args:
        url: URL to validate.
        allow_private: If True, allow private/internal addresses.

    Raises:
        URLValidationError: If URL validation fails.
    """
    is_valid, error = validate_url(url, allow_private)
    if not is_valid:
        raise URLValidationError(url, error)


# =============================================================================
# HTTP Cache
# =============================================================================

class HTTPCache:
    """Manages HTTP response caching."""

    def __init__(self, cache_dir: str):
        """Initialize the HTTP cache.

        Args:
            cache_dir: Directory to store cache files.
        """
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"HTTP cache enabled: {cache_dir}")

    def _get_cache_path(self, url: str) -> str:
        """Get cache file path for a URL.

        Args:
            url: URL to get cache path for.

        Returns:
            Path to cache file.
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.json")

    def get(self, url: str) -> Optional[Tuple[str, int]]:
        """Get cached response for a URL.

        Args:
            url: URL to get cached response for.

        Returns:
            Tuple of (content, status_code) if cached and valid, None otherwise.
            Corrupted or invalid cache files are cleaned up and None is returned.
        """
        cache_path = self._get_cache_path(url)

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Validate JSON before parsing
            if not content.strip():
                logger.warning(f"Empty cache file for {url}, removing")
                self._remove_cache_file(cache_path)
                return None

            cached = json.loads(content)

            # Validate required fields
            if not isinstance(cached, dict):
                logger.warning(f"Cache file for {url} is not a JSON object, removing")
                self._remove_cache_file(cache_path)
                return None

            if 'content' not in cached or 'status' not in cached:
                logger.warning(f"Cache file for {url} missing required fields, removing")
                self._remove_cache_file(cache_path)
                return None

            logger.info(f"Cache hit: {url}")
            return cached['content'], cached['status']

        except json.JSONDecodeError as e:
            # Cache file is corrupted - remove it and return None
            logger.warning(f"Corrupted cache file for {url}: {e}")
            self._remove_cache_file(cache_path)
            return None

        except OSError as e:
            logger.warning(f"Error reading cache file for {url}: {e}")
            return None

    def _remove_cache_file(self, cache_path: str) -> None:
        """Remove a cache file safely.

        Args:
            cache_path: Path to cache file to remove.
        """
        try:
            os.remove(cache_path)
            logger.info(f"Removed invalid cache file: {cache_path}")
        except OSError:
            pass

    def put(self, url: str, content: str, status: int) -> None:
        """Store response in cache.

        Args:
            url: URL that was fetched.
            content: Response content.
            status: HTTP status code.
        """
        cache_path = self._get_cache_path(url)

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({'url': url, 'content': content, 'status': status}, f)
            logger.info(f"Cached: {url}")
        except OSError as e:
            logger.warning(f"Failed to cache {url}: {e}")


# =============================================================================
# URL Utilities
# =============================================================================

def normalize_url(url: str, preserve_trailing_slash: bool = False) -> str:
    """Normalize a URL by removing fragments.

    Args:
        url: URL to normalize.
        preserve_trailing_slash: If True, keep trailing slashes for directory-like
            paths (paths without file extensions). This is important for correct
            relative URL resolution.

    Returns:
        Normalized URL.
    """
    parsed = urllib.parse.urlparse(url)
    # Remove fragment
    normalized = parsed._replace(fragment='')
    # Rebuild URL
    result = urllib.parse.urlunparse(normalized)

    # Handle trailing slash
    if result.endswith('/') and parsed.path != '/':
        if preserve_trailing_slash:
            # Keep trailing slash for directory-like paths (no file extension)
            path = parsed.path.rstrip('/')
            has_extension = '.' in path.split('/')[-1] if path else False
            if has_extension:
                result = result.rstrip('/')
            # else: keep the trailing slash for directories
        else:
            result = result.rstrip('/')

    return result


def is_same_domain(url: str, allowed_domains: List[str]) -> bool:
    """Check if a URL belongs to one of the allowed domains.

    Args:
        url: URL to check.
        allowed_domains: List of allowed domain names.

    Returns:
        True if URL is in an allowed domain.
    """
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc in allowed_domains


def is_excluded_path(url: str, excluded_paths: List[str]) -> bool:
    """Check if a URL path matches any excluded path prefix.

    Args:
        url: URL to check.
        excluded_paths: List of path prefixes to exclude.

    Returns:
        True if URL path starts with any excluded prefix.
    """
    parsed = urllib.parse.urlparse(url)
    for excluded in excluded_paths:
        if parsed.path.startswith(excluded):
            return True
    return False


def is_included_path(url: str, included_paths: Optional[List[str]]) -> bool:
    """Check if a URL path matches any included path prefix.

    Args:
        url: URL to check.
        included_paths: List of path prefixes to include. If None, returns True.

    Returns:
        True if included_paths is None OR URL path starts with any included prefix.
    """
    if included_paths is None:
        return True

    parsed = urllib.parse.urlparse(url)
    for included in included_paths:
        if parsed.path.startswith(included):
            return True
    return False


def _ensure_directory_slash(url: str) -> str:
    """Ensure directory-like URLs have a trailing slash for proper relative resolution.

    Args:
        url: URL to check.

    Returns:
        URL with trailing slash added if it looks like a directory.
    """
    # Common web file extensions
    FILE_EXTENSIONS = {
        'html', 'htm', 'php', 'asp', 'aspx', 'jsp', 'cgi',
        'xml', 'json', 'txt', 'css', 'js',
        'pdf', 'doc', 'docx', 'xls', 'xlsx',
        'png', 'jpg', 'jpeg', 'gif', 'svg', 'ico',
        'zip', 'tar', 'gz', 'rar',
        'mp3', 'mp4', 'wav', 'avi',
    }

    parsed = urllib.parse.urlparse(url)
    path = parsed.path

    # If path is empty or root, return as-is
    if not path or path == '/':
        return url

    # Check if path looks like a file (has known web file extension)
    last_segment = path.rstrip('/').split('/')[-1]
    has_file_extension = False
    if '.' in last_segment:
        ext = last_segment.rsplit('.', 1)[-1].lower()
        has_file_extension = ext in FILE_EXTENSIONS

    # If it's a directory-like path without trailing slash, add one
    if not has_file_extension and not path.endswith('/'):
        new_path = path + '/'
        parsed = parsed._replace(path=new_path)
        return urllib.parse.urlunparse(parsed)

    return url


class LinkExtractor(HTMLParser):
    """HTML parser to extract links from anchor tags."""

    def __init__(self, base_url: str):
        super().__init__()
        # Ensure base URL has trailing slash for directories for proper relative resolution
        self.base_url = _ensure_directory_slash(base_url)
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        if tag == 'a':
            for name, value in attrs:
                if name == 'href' and value:
                    # Skip fragment-only links
                    if value.startswith('#'):
                        continue
                    # Resolve relative URLs
                    absolute_url = urllib.parse.urljoin(self.base_url, value)
                    # Normalize (don't strip trailing slash for dedup purposes)
                    normalized = normalize_url(absolute_url)
                    self.links.append(normalized)


def extract_links(html: str, base_url: str) -> List[str]:
    """Extract all links from HTML content.

    Args:
        html: HTML content.
        base_url: Base URL for resolving relative links. Directory-like URLs
            will automatically get a trailing slash for correct resolution.

    Returns:
        List of absolute URLs found in the HTML.
    """
    parser = LinkExtractor(base_url)
    try:
        parser.feed(html)
    except (ValueError, AssertionError):
        # HTMLParser can raise these on malformed HTML
        pass
    return parser.links


# =============================================================================
# Robots.txt Parser
# =============================================================================

class RobotsParser:
    """Simple robots.txt parser."""

    def __init__(self, base_url: str, robots_txt: Optional[str]):
        """Initialize the parser.

        Args:
            base_url: Base URL of the site.
            robots_txt: Contents of robots.txt, or None if not available.
        """
        self.base_url = base_url
        self.disallowed: List[str] = []
        self.allowed: List[str] = []
        self.crawl_delay: Optional[float] = None

        if robots_txt:
            self._parse(robots_txt)

    def _parse(self, content: str) -> None:
        """Parse robots.txt content.

        Args:
            content: robots.txt file content.
        """
        in_user_agent_block = False

        for line in content.split('\n'):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse directives
            if ':' in line:
                directive, value = line.split(':', 1)
                directive = directive.strip().lower()
                value = value.strip()

                if directive == 'user-agent':
                    # We're interested in * or our user agent
                    in_user_agent_block = value == '*'
                elif in_user_agent_block:
                    if directive == 'disallow' and value:
                        self.disallowed.append(value)
                    elif directive == 'allow' and value:
                        self.allowed.append(value)
                    elif directive == 'crawl-delay':
                        try:
                            self.crawl_delay = float(value)
                        except ValueError:
                            pass

    def can_fetch(self, path: str) -> bool:
        """Check if a path can be fetched according to robots.txt.

        Args:
            path: URL path to check.

        Returns:
            True if path is allowed.
        """
        # If no rules, allow all
        if not self.disallowed and not self.allowed:
            return True

        # Check allow rules first (they take precedence)
        for allowed in self.allowed:
            if path.startswith(allowed):
                return True

        # Check disallow rules
        for disallowed in self.disallowed:
            if path.startswith(disallowed):
                return False

        return True

    def get_crawl_delay(self) -> Optional[float]:
        """Get the crawl delay from robots.txt.

        Returns:
            Crawl delay in seconds, or None if not specified.
        """
        return self.crawl_delay


# =============================================================================
# Crawler Class
# =============================================================================

class Crawler:
    """BFS web crawler with rate limiting and domain filtering."""

    USER_AGENT = 'RAGSystemCrawler/1.0'

    def __init__(self, start_url: str, allowed_domains: List[str],
                 excluded_paths: Optional[List[str]] = None,
                 included_paths: Optional[List[str]] = None,
                 max_pages: int = 1000, delay: float = 1.0,
                 cache_dir: Optional[str] = None,
                 ignore_robots: bool = False,
                 allow_private_urls: bool = False):
        """Initialize the crawler.

        Args:
            start_url: Starting URL to crawl.
            allowed_domains: List of domains to stay within.
            excluded_paths: List of path prefixes to exclude.
            included_paths: List of path prefixes to include. If set, only URLs
                          whose path starts with one of these prefixes will be crawled.
                          If None, all paths are included (unless excluded).
            max_pages: Maximum number of pages to crawl.
            delay: Delay between requests in seconds.
            cache_dir: Optional directory for HTTP response caching.
                      If None, caching is disabled.
            ignore_robots: If True, ignore robots.txt restrictions.
            allow_private_urls: If True, allow crawling private/internal URLs.
        """
        self.start_url = normalize_url(start_url)
        self.allowed_domains = allowed_domains
        self.excluded_paths = excluded_paths or []
        self.included_paths = included_paths
        self.max_pages = max_pages
        self.delay = delay
        self.ignore_robots = ignore_robots
        self.allow_private_urls = allow_private_urls

        # Set up HTTP cache if directory specified
        self.cache: Optional[HTTPCache] = None
        if cache_dir:
            self.cache = HTTPCache(cache_dir)

        if self.ignore_robots:
            logger.info("Ignoring robots.txt restrictions")

        self.visited: Set[str] = set()
        self.queue: List[str] = [self.start_url]
        self.robots_parser: Optional[RobotsParser] = None
        self.stats: CrawlStats = CrawlStats()

    def _fetch_robots_txt(self) -> Optional[str]:
        """Fetch and parse robots.txt for the start domain.

        Returns:
            robots.txt content or None.
        """
        parsed = urllib.parse.urlparse(self.start_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        try:
            request = urllib.request.Request(
                robots_url,
                headers={'User-Agent': self.USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.read().decode('utf-8', errors='ignore')
        except urllib.error.HTTPError as e:
            logger.debug(f"robots.txt fetch failed with HTTP {e.code}")
            return None
        except urllib.error.URLError as e:
            logger.debug(f"robots.txt fetch failed: {e.reason}")
            return None
        except OSError as e:
            logger.debug(f"robots.txt fetch failed with OS error: {e}")
            return None

    def _fetch_url_from_network(self, url: str) -> Tuple[str, int]:
        """Fetch a URL directly from the network.

        Args:
            url: URL to fetch.

        Returns:
            Tuple of (content, status_code).

        Raises:
            urllib.error.HTTPError: If the request fails.
            urllib.error.URLError: If the connection fails.
        """
        request = urllib.request.Request(
            url,
            headers={
                'User-Agent': self.USER_AGENT,
                'Accept': 'text/html,application/xhtml+xml',
            }
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read().decode('utf-8', errors='replace')
            status = response.status
        return content, status

    def _fetch_url(self, url: str) -> Tuple[str, int]:
        """Fetch a URL and return its content.

        Checks cache first if caching is enabled. Retries on transient errors
        (5xx status codes and timeouts) with exponential backoff.

        Args:
            url: URL to fetch.

        Returns:
            Tuple of (content, status_code).

        Raises:
            FetchError: If the request fails after all retries.
            URLValidationError: If the URL fails validation.
        """
        # Validate URL before fetching
        validate_url_or_raise(url, allow_private=self.allow_private_urls)

        # Check cache first
        if self.cache:
            cached = self.cache.get(url)
            if cached is not None:
                return cached

        # Fetch from network with retry logic
        max_retries = config.CRAWLER_MAX_RETRIES
        base_delay = config.CRAWLER_RETRY_DELAY
        retry_codes = config.CRAWLER_RETRY_STATUS_CODES

        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                content, status = self._fetch_url_from_network(url)

                # Cache the response
                if self.cache:
                    self.cache.put(url, content, status)

                return content, status

            except urllib.error.HTTPError as e:
                last_error = e
                # Only retry on configured status codes
                should_retry = e.code in retry_codes and attempt < max_retries
                if not should_retry:
                    raise FetchError(
                        url=url,
                        message=f"HTTP {e.code}: {e.reason}",
                        status_code=e.code
                    )

                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"HTTP {e.code} for {url}, retrying in {delay}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)

            except urllib.error.URLError as e:
                last_error = e
                # Retry on timeout and connection errors
                if attempt >= max_retries:
                    raise FetchError(
                        url=url,
                        message=f"Connection error: {e.reason}"
                    )

                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"URL error for {url}: {e.reason}, retrying in {delay}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)

            except OSError as e:
                last_error = e
                # Retry on OS-level errors (socket errors, etc.)
                if attempt >= max_retries:
                    raise FetchError(
                        url=url,
                        message=f"OS error: {e}"
                    )

                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"OS error for {url}: {e}, retrying in {delay}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)

        # Should not reach here, but just in case
        raise FetchError(url=url, message=f"Max retries exceeded: {last_error}")

    def _is_html_content(self, url: str) -> bool:
        """Check if URL likely points to HTML content.

        Args:
            url: URL to check.

        Returns:
            True if URL appears to be HTML.
        """
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.lower()

        # Check for non-HTML extensions
        non_html_extensions = {
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.tar', '.gz', '.rar', '.7z',
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
            '.mp3', '.mp4', '.wav', '.avi', '.mov',
            '.css', '.js', '.json', '.xml', '.csv',
        }

        for ext in non_html_extensions:
            if path.endswith(ext):
                return False

        return True

    def _should_crawl(self, url: str) -> bool:
        """Check if a URL should be crawled.

        Args:
            url: URL to check.

        Returns:
            True if URL should be crawled.
        """
        # Already visited?
        if url in self.visited:
            return False

        # Same domain?
        if not is_same_domain(url, self.allowed_domains):
            return False

        # Excluded path?
        if is_excluded_path(url, self.excluded_paths):
            return False

        # Included path?
        if not is_included_path(url, self.included_paths):
            return False

        # HTML content?
        if not self._is_html_content(url):
            return False

        # Robots.txt allows?
        if self.robots_parser:
            parsed = urllib.parse.urlparse(url)
            if not self.robots_parser.can_fetch(parsed.path):
                return False

        return True

    def crawl(self) -> Generator[Dict, None, CrawlStats]:
        """Crawl the website starting from the start URL.

        Yields:
            Dict with 'url', 'html', and 'status_code' for each page.

        Returns:
            CrawlStats object with crawl statistics (access via get_stats()).
        """
        # Reset stats for new crawl
        self.stats = CrawlStats()

        # Fetch robots.txt first (unless ignoring)
        if not self.ignore_robots:
            robots_txt = self._fetch_robots_txt()
            self.robots_parser = RobotsParser(self.start_url, robots_txt)

            # Use crawl delay from robots.txt if specified
            if self.robots_parser.get_crawl_delay():
                self.delay = max(self.delay, self.robots_parser.get_crawl_delay())

        pages_crawled = 0

        while self.queue and pages_crawled < self.max_pages:
            # Check for shutdown request
            if is_shutdown_requested():
                logger.info("Shutdown requested, stopping crawl...")
                break

            url = self.queue.pop(0)

            # Skip if already visited or shouldn't crawl
            if url in self.visited:
                continue

            if not self._should_crawl(url):
                self.visited.add(url)
                self.stats.record_skip()
                continue

            self.visited.add(url)

            try:
                html, status_code = self._fetch_url(url)
                pages_crawled += 1
                self.stats.record_success()

                logger.info(f"Crawled ({pages_crawled}/{self.max_pages}): {url}")

                # Extract and queue new links
                links = extract_links(html, url)
                for link in links:
                    if link not in self.visited and self._should_crawl(link):
                        self.queue.append(link)

                yield {
                    'url': url,
                    'html': html,
                    'status_code': status_code
                }

                # Rate limiting
                if self.delay > 0 and pages_crawled < self.max_pages:
                    time.sleep(self.delay)

            except FetchError as e:
                error_msg = str(e)
                logger.warning(f"Fetch error for {url}: {error_msg}")
                self.stats.record_failure(url, error_msg)

            except URLValidationError as e:
                error_msg = str(e)
                logger.warning(f"URL validation error: {error_msg}")
                self.stats.record_failure(url, error_msg)

        return self.stats

    def get_stats(self) -> CrawlStats:
        """Get the current crawl statistics.

        Returns:
            CrawlStats object with crawl statistics.
        """
        return self.stats
