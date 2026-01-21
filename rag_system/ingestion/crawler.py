"""Web crawler module for the RAG system.

Provides BFS web crawling with robots.txt respect, rate limiting,
and domain filtering. Uses only Python standard library.
"""

import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Dict, Generator, List, Optional, Set

from rag_system.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# URL Utilities
# =============================================================================

def normalize_url(url: str) -> str:
    """Normalize a URL by removing fragments and trailing slashes.

    Args:
        url: URL to normalize.

    Returns:
        Normalized URL.
    """
    parsed = urllib.parse.urlparse(url)
    # Remove fragment
    normalized = parsed._replace(fragment='')
    # Rebuild URL
    result = urllib.parse.urlunparse(normalized)
    # Remove trailing slash (except for root)
    if result.endswith('/') and parsed.path != '/':
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


class LinkExtractor(HTMLParser):
    """HTML parser to extract links from anchor tags."""

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
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
                    # Normalize
                    normalized = normalize_url(absolute_url)
                    self.links.append(normalized)


def extract_links(html: str, base_url: str) -> List[str]:
    """Extract all links from HTML content.

    Args:
        html: HTML content.
        base_url: Base URL for resolving relative links.

    Returns:
        List of absolute URLs found in the HTML.
    """
    parser = LinkExtractor(base_url)
    try:
        parser.feed(html)
    except Exception:
        pass  # Ignore malformed HTML
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
                 max_pages: int = 1000, delay: float = 1.0):
        """Initialize the crawler.

        Args:
            start_url: Starting URL to crawl.
            allowed_domains: List of domains to stay within.
            excluded_paths: List of path prefixes to exclude.
            max_pages: Maximum number of pages to crawl.
            delay: Delay between requests in seconds.
        """
        self.start_url = normalize_url(start_url)
        self.allowed_domains = allowed_domains
        self.excluded_paths = excluded_paths or []
        self.max_pages = max_pages
        self.delay = delay

        self.visited: Set[str] = set()
        self.queue: List[str] = [self.start_url]
        self.robots_parser: Optional[RobotsParser] = None

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
        except Exception:
            return None

    def _fetch_url(self, url: str) -> tuple:
        """Fetch a URL and return its content.

        Args:
            url: URL to fetch.

        Returns:
            Tuple of (content, status_code).

        Raises:
            urllib.error.HTTPError: If the request fails.
        """
        request = urllib.request.Request(
            url,
            headers={
                'User-Agent': self.USER_AGENT,
                'Accept': 'text/html,application/xhtml+xml',
            }
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read().decode('utf-8', errors='ignore')
            return content, response.status

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

        # HTML content?
        if not self._is_html_content(url):
            return False

        # Robots.txt allows?
        if self.robots_parser:
            parsed = urllib.parse.urlparse(url)
            if not self.robots_parser.can_fetch(parsed.path):
                return False

        return True

    def crawl(self) -> Generator[Dict, None, None]:
        """Crawl the website starting from the start URL.

        Yields:
            Dict with 'url', 'html', and 'status_code' for each page.
        """
        # Fetch robots.txt first
        robots_txt = self._fetch_robots_txt()
        self.robots_parser = RobotsParser(self.start_url, robots_txt)

        # Use crawl delay from robots.txt if specified
        if self.robots_parser.get_crawl_delay():
            self.delay = max(self.delay, self.robots_parser.get_crawl_delay())

        pages_crawled = 0

        while self.queue and pages_crawled < self.max_pages:
            url = self.queue.pop(0)

            # Skip if already visited or shouldn't crawl
            if url in self.visited:
                continue

            if not self._should_crawl(url):
                self.visited.add(url)
                continue

            self.visited.add(url)

            try:
                html, status_code = self._fetch_url(url)
                pages_crawled += 1

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

            except urllib.error.HTTPError as e:
                logger.warning(f"HTTP error {e.code} for {url}")
            except urllib.error.URLError as e:
                logger.warning(f"URL error for {url}: {e.reason}")
            except Exception as e:
                logger.warning(f"Error crawling {url}: {e}")
