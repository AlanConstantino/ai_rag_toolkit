"""HTML parser module for the RAG system.

Extracts text, headings, and metadata from HTML documents.
Uses only Python standard library (html.parser).
"""

import re
from html.parser import HTMLParser
from typing import Dict, List, Optional, Any


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

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        tag = tag.lower()
        self.current_tag_stack.append(tag)

        # Check if we should ignore this tag's content
        if tag in self.IGNORE_TAGS:
            self.ignore_depth += 1
        elif tag == 'nav' and self.remove_nav:
            self.ignore_depth += 1
        elif tag == 'footer' and self.remove_footer:
            self.ignore_depth += 1
        elif tag == 'header' and self.remove_nav:
            self.ignore_depth += 1

        # Add spacing for block elements
        if tag in self.BLOCK_TAGS and self.text_parts:
            self.text_parts.append('\n')

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        # Pop from stack (handle malformed HTML)
        if self.current_tag_stack and self.current_tag_stack[-1] == tag:
            self.current_tag_stack.pop()

        # Decrease ignore depth if we're leaving an ignored tag
        if tag in self.IGNORE_TAGS:
            self.ignore_depth = max(0, self.ignore_depth - 1)
        elif tag == 'nav' and self.remove_nav:
            self.ignore_depth = max(0, self.ignore_depth - 1)
        elif tag == 'footer' and self.remove_footer:
            self.ignore_depth = max(0, self.ignore_depth - 1)
        elif tag == 'header' and self.remove_nav:
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
    """HTML parser that extracts headings."""

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
# Public Functions
# =============================================================================

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
