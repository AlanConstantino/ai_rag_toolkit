"""HTML parser module for the RAG system.

Extracts text, headings, and metadata from HTML documents.
Uses only Python standard library (html.parser).
"""

import re
from html.parser import HTMLParser
from typing import Dict, List, Optional, Any

from rag_system import config


# =============================================================================
# HTML Text Extractor
# =============================================================================

class TextExtractor(HTMLParser):
    """HTML parser that extracts readable text content."""

    # Tags whose content should be completely ignored
    IGNORE_TAGS = {'script', 'style', 'noscript', 'template'}

    # Tags that should optionally be removed
    OPTIONAL_IGNORE_TAGS = {'nav', 'footer', 'header', 'aside'}

    # Block-level tags that should add newlines
    BLOCK_TAGS = {
        'p', 'div', 'section', 'article', 'main', 'aside',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'dl', 'dt', 'dd',
        'table', 'tr', 'th', 'td',
        'blockquote', 'pre', 'code',
        'br', 'hr'
    }

    def __init__(self, remove_nav: bool = False, remove_footer: bool = False):
        super().__init__()
        self.text_parts: List[str] = []
        self.ignore_depth = 0
        self.remove_nav = remove_nav
        self.remove_footer = remove_footer
        self.current_tag_stack: List[str] = []

    def _should_ignore_tag(self, tag: str) -> bool:
        """Check if a tag should be ignored based on settings."""
        if tag in self.IGNORE_TAGS:
            return True
        if self.remove_nav and tag in ('nav', 'header'):
            return True
        if self.remove_footer and tag == 'footer':
            return True
        return False

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        tag = tag.lower()
        self.current_tag_stack.append(tag)

        if self._should_ignore_tag(tag):
            self.ignore_depth += 1

        # Add spacing for block elements
        if tag in self.BLOCK_TAGS and self.text_parts:
            self.text_parts.append('\n')

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        # Pop from stack (handle malformed HTML)
        if self.current_tag_stack and self.current_tag_stack[-1] == tag:
            self.current_tag_stack.pop()

        if self._should_ignore_tag(tag):
            self.ignore_depth = max(0, self.ignore_depth - 1)

        # Add spacing after block elements
        if tag in self.BLOCK_TAGS:
            self.text_parts.append('\n')

    def handle_data(self, data: str) -> None:
        if self.ignore_depth == 0:
            # Clean up the text but preserve it
            text = data.strip()
            if text:
                self.text_parts.append(text)

    def get_text(self) -> str:
        """Get the extracted text, normalized."""
        text = ' '.join(self.text_parts)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()


# =============================================================================
# Heading Extractor
# =============================================================================

class HeadingExtractor(HTMLParser):
    """HTML parser that extracts headings.

    Note: This extracts ALL headings from HTML without filtering.
    For chunking, we use the Markdown-first approach (html_to_markdown.py)
    which naturally filters out navigation/sidebar headings by only
    converting content structure to Markdown.
    """

    HEADING_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}

    def __init__(self):
        super().__init__()
        self.headings: List[Dict[str, Any]] = []
        self.current_heading: Optional[Dict[str, Any]] = None
        self.current_text: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        tag = tag.lower()
        if tag in self.HEADING_TAGS:
            level = int(tag[1])
            self.current_heading = {'level': level, 'text': ''}
            self.current_text = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.HEADING_TAGS and self.current_heading:
            self.current_heading['text'] = ' '.join(self.current_text).strip()
            # Add all non-empty headings (no filtering - let Markdown conversion handle it)
            if self.current_heading['text']:
                self.headings.append(self.current_heading)
            self.current_heading = None
            self.current_text = []

    def handle_data(self, data: str) -> None:
        if self.current_heading is not None:
            text = data.strip()
            if text:
                self.current_text.append(text)


# =============================================================================
# Title Extractor
# =============================================================================

class TitleExtractor(HTMLParser):
    """HTML parser that extracts the title."""

    def __init__(self):
        super().__init__()
        self.title = ''
        self.in_title = False
        self.title_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        if tag.lower() == 'title':
            self.in_title = True
            self.title_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == 'title':
            self.in_title = False
            self.title = ' '.join(self.title_parts).strip()

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)


# =============================================================================
# Metadata Extractor
# =============================================================================

class MetadataExtractor(HTMLParser):
    """HTML parser that extracts metadata from meta tags."""

    def __init__(self):
        super().__init__()
        self.metadata: Dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        if tag.lower() == 'meta':
            attrs_dict = dict(attrs)

            # Handle name-based meta tags
            name = attrs_dict.get('name', '').lower()
            content = attrs_dict.get('content', '')

            if name and content:
                self.metadata[name] = content

            # Handle property-based meta tags (OpenGraph, etc.)
            prop = attrs_dict.get('property', '')
            if prop and content:
                self.metadata[prop] = content


# =============================================================================
# Main Content Detector
# =============================================================================

def _matches_selector(selector: str, tag: str, attrs: List[tuple]) -> bool:
    """Check if a tag matches a selector pattern.

    Args:
        selector: Selector string (e.g., 'main', 'role=main', 'class=content')
        tag: HTML tag name
        attrs: List of tag attributes

    Returns:
        True if the tag matches the selector.
    """
    tag = tag.lower()
    attrs_dict = dict(attrs)

    # Simple tag name match
    if selector == tag:
        return True

    # Attribute-based matches
    if '=' not in selector:
        return False

    attr_name, expected_value = selector.split('=', 1)

    if attr_name == 'role':
        return attrs_dict.get('role', '').lower() == expected_value
    elif attr_name == 'class':
        classes = attrs_dict.get('class', '').lower().split()
        return expected_value in classes
    elif attr_name == 'id':
        return attrs_dict.get('id', '').lower() == expected_value

    return False


class MainContentFinder(HTMLParser):
    """HTML parser that identifies the main content element type.

    Searches for semantic HTML5 tags (<main>, <article>), ARIA roles,
    and common class/id patterns that indicate main content areas.
    """

    def __init__(self, selectors: Optional[List[str]] = None):
        super().__init__()
        self.selectors = selectors or config.MAIN_CONTENT_SELECTORS
        self.found_element: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        if self.found_element:
            return

        for selector in self.selectors:
            if _matches_selector(selector, tag, attrs):
                self.found_element = selector
                return


class MainContentExtractor(HTMLParser):
    """HTML parser that extracts content from the main content area only.

    Finds the main content element and extracts text from within it,
    excluding navigation, sidebars, and footers.
    """

    IGNORE_TAGS = {'script', 'style', 'noscript', 'template', 'nav', 'footer', 'header', 'aside'}
    BLOCK_TAGS = {
        'p', 'div', 'section', 'article', 'main',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'dl', 'dt', 'dd',
        'table', 'tr', 'th', 'td',
        'blockquote', 'pre', 'code',
        'br', 'hr'
    }

    def __init__(self, target_selector: str):
        super().__init__()
        self.target_selector = target_selector
        self.text_parts: List[str] = []
        self.in_main_content = False
        self.main_content_depth = 0
        self.ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        tag_lower = tag.lower()

        # Check if entering the main content element
        if not self.in_main_content and _matches_selector(self.target_selector, tag, attrs):
            self.in_main_content = True
            self.main_content_depth = 1
            return

        if self.in_main_content:
            self.main_content_depth += 1

            if tag_lower in self.IGNORE_TAGS:
                self.ignore_depth += 1

            # Add spacing for block elements
            if tag_lower in self.BLOCK_TAGS and self.text_parts:
                self.text_parts.append('\n')

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()

        if self.in_main_content:
            if tag_lower in self.IGNORE_TAGS:
                self.ignore_depth = max(0, self.ignore_depth - 1)

            # Add spacing after block elements
            if tag_lower in self.BLOCK_TAGS:
                self.text_parts.append('\n')

            self.main_content_depth -= 1
            if self.main_content_depth <= 0:
                self.in_main_content = False

    def handle_data(self, data: str) -> None:
        if self.in_main_content and self.ignore_depth == 0:
            text = data.strip()
            if text:
                self.text_parts.append(text)

    def get_text(self) -> str:
        """Get the extracted main content text, normalized."""
        text = ' '.join(self.text_parts)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()


# =============================================================================
# Public Functions
# =============================================================================

def find_main_content_element(html: str) -> Optional[str]:
    """Find the type of main content element in HTML.

    Searches for semantic HTML5 tags, ARIA roles, and common class/id patterns.

    Args:
        html: HTML content.

    Returns:
        Selector string (e.g., 'main', 'article', 'class=main-content') or None.
    """
    parser = MainContentFinder()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.found_element


def extract_main_content(html: str) -> str:
    """Extract text from the main content area of HTML.

    Identifies the main content element (using <main>, <article>, role='main',
    or common class/id patterns) and extracts text only from within it.
    Falls back to full content extraction if no main content area is found.

    Args:
        html: HTML content.

    Returns:
        Extracted main content text.
    """
    # First, find the main content element
    main_element = find_main_content_element(html)

    if main_element:
        # Extract content from within the main element
        parser = MainContentExtractor(main_element)
        try:
            parser.feed(html)
        except Exception:
            pass
        content = parser.get_text()
        if content:
            return content

    # Fall back to full content extraction (minus nav/footer)
    return extract_text(html, remove_nav=True, remove_footer=True)


def extract_title(html: str) -> str:
    """Extract the page title from HTML.

    Args:
        html: HTML content.

    Returns:
        Page title, or empty string if not found.
    """
    parser = TitleExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.title


def extract_text(html: str, remove_nav: bool = False,
                 remove_footer: bool = False) -> str:
    """Extract readable text from HTML.

    Args:
        html: HTML content.
        remove_nav: Whether to remove navigation elements.
        remove_footer: Whether to remove footer elements.

    Returns:
        Extracted text content.
    """
    parser = TextExtractor(remove_nav=remove_nav, remove_footer=remove_footer)
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.get_text()


def extract_headings(html: str) -> List[Dict[str, Any]]:
    """Extract headings from HTML.

    Args:
        html: HTML content.

    Returns:
        List of heading dicts with 'level' and 'text' keys.
    """
    parser = HeadingExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.headings


def extract_metadata(html: str) -> Dict[str, str]:
    """Extract metadata from HTML meta tags.

    Args:
        html: HTML content.

    Returns:
        Dict of metadata key-value pairs.
    """
    parser = MetadataExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.metadata


def parse_html(html: str, remove_nav: bool = False,
               remove_footer: bool = False) -> Dict[str, Any]:
    """Parse HTML and extract all relevant content.

    Args:
        html: HTML content.
        remove_nav: Whether to remove navigation elements.
        remove_footer: Whether to remove footer elements.

    Returns:
        Dict with 'title', 'text', 'headings', and 'metadata' keys.
    """
    return {
        'title': extract_title(html),
        'text': extract_text(html, remove_nav=remove_nav,
                            remove_footer=remove_footer),
        'headings': extract_headings(html),
        'metadata': extract_metadata(html)
    }
