"""HTML to Markdown converter for the RAG system.

Converts HTML content to clean Markdown, preserving semantic structure
while stripping navigation, sidebars, and other non-content elements.
Uses only Python standard library.
"""

import re
from html.parser import HTMLParser
from typing import List, Optional, Tuple


class HTMLToMarkdownConverter(HTMLParser):
    """Converts HTML to Markdown, focusing on content structure."""

    # Tags that indicate non-content areas to skip entirely
    SKIP_TAGS = {'nav', 'aside', 'footer', 'header', 'script', 'style', 'noscript', 'template'}

    # Block-level tags that need newlines
    BLOCK_TAGS = {
        'p', 'div', 'section', 'article', 'main',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'dl', 'dt', 'dd',
        'blockquote', 'pre', 'table', 'tr', 'th', 'td',
        'br', 'hr'
    }

    # Heading tags mapped to markdown prefix
    HEADING_MAP = {
        'h1': '#',
        'h2': '##',
        'h3': '###',
        'h4': '####',
        'h5': '#####',
        'h6': '######'
    }

    def __init__(self):
        super().__init__()
        self.output: List[str] = []
        self.skip_depth = 0  # Track nested skip tags
        self.tag_stack: List[str] = []
        self.in_pre = False
        self.in_code = False
        self.in_list = False
        self.list_type_stack: List[str] = []  # 'ul' or 'ol'
        self.list_item_count_stack: List[int] = []
        self.current_link_href: Optional[str] = None
        self.link_text: List[str] = []
        self.in_link = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        self.tag_stack.append(tag)
        attrs_dict = dict(attrs)

        # Skip non-content regions
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return

        if self.skip_depth > 0:
            return

        # Handle specific tags
        if tag in self.HEADING_MAP:
            self._add_newlines(2)
            self.output.append(self.HEADING_MAP[tag] + ' ')

        elif tag == 'p':
            self._add_newlines(2)

        elif tag == 'br':
            self.output.append('\n')

        elif tag == 'hr':
            self._add_newlines(2)
            self.output.append('---')
            self._add_newlines(2)

        elif tag == 'strong' or tag == 'b':
            self.output.append('**')

        elif tag == 'em' or tag == 'i':
            self.output.append('*')

        elif tag == 'code':
            if not self.in_pre:
                self.output.append('`')
            self.in_code = True

        elif tag == 'pre':
            self._add_newlines(2)
            self.output.append('```\n')
            self.in_pre = True

        elif tag == 'blockquote':
            self._add_newlines(2)
            self.output.append('> ')

        elif tag == 'ul':
            self._add_newlines(1)
            self.list_type_stack.append('ul')
            self.list_item_count_stack.append(0)
            self.in_list = True

        elif tag == 'ol':
            self._add_newlines(1)
            self.list_type_stack.append('ol')
            self.list_item_count_stack.append(0)
            self.in_list = True

        elif tag == 'li':
            self._add_newlines(1)
            indent = '  ' * (len(self.list_type_stack) - 1)
            if self.list_type_stack and self.list_type_stack[-1] == 'ol':
                self.list_item_count_stack[-1] += 1
                self.output.append(f'{indent}{self.list_item_count_stack[-1]}. ')
            else:
                self.output.append(f'{indent}- ')

        elif tag == 'a':
            href = attrs_dict.get('href', '')
            if href and not href.startswith('#'):
                self.current_link_href = href
                self.in_link = True
                self.link_text = []

        elif tag == 'img':
            alt = attrs_dict.get('alt', '')
            src = attrs_dict.get('src', '')
            if src:
                self.output.append(f'![{alt}]({src})')

        elif tag == 'table':
            self._add_newlines(2)

        elif tag == 'tr':
            self._add_newlines(1)
            self.output.append('| ')

        elif tag == 'th' or tag == 'td':
            pass  # Content handled in data

        elif tag in self.BLOCK_TAGS:
            self._add_newlines(1)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        # Pop from stack
        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()

        # Handle skip tags
        if tag in self.SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
            return

        if self.skip_depth > 0:
            return

        # Handle specific closing tags
        if tag in self.HEADING_MAP:
            self._add_newlines(2)

        elif tag == 'p':
            self._add_newlines(2)

        elif tag == 'strong' or tag == 'b':
            self.output.append('**')

        elif tag == 'em' or tag == 'i':
            self.output.append('*')

        elif tag == 'code':
            if not self.in_pre:
                self.output.append('`')
            self.in_code = False

        elif tag == 'pre':
            self.output.append('\n```')
            self._add_newlines(2)
            self.in_pre = False

        elif tag == 'blockquote':
            self._add_newlines(2)

        elif tag == 'ul' or tag == 'ol':
            if self.list_type_stack:
                self.list_type_stack.pop()
            if self.list_item_count_stack:
                self.list_item_count_stack.pop()
            self.in_list = len(self.list_type_stack) > 0
            self._add_newlines(1)

        elif tag == 'a':
            if self.in_link and self.current_link_href:
                link_text = ''.join(self.link_text).strip()
                if link_text:
                    self.output.append(f'[{link_text}]({self.current_link_href})')
            self.in_link = False
            self.current_link_href = None
            self.link_text = []

        elif tag == 'th' or tag == 'td':
            self.output.append(' | ')

        elif tag == 'tr':
            pass  # Row ended

        elif tag == 'thead':
            # Add markdown table header separator
            self._add_newlines(1)
            self.output.append('|---')

    def handle_data(self, data: str) -> None:
        if self.skip_depth > 0:
            return

        # Clean the text
        if self.in_pre:
            # Preserve whitespace in preformatted blocks
            self.output.append(data)
        else:
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', data)
            if text.strip():
                if self.in_link:
                    self.link_text.append(text)
                else:
                    self.output.append(text)

    def _add_newlines(self, count: int) -> None:
        """Add newlines, avoiding excessive blank lines."""
        if not self.output:
            return

        # Count existing trailing newlines
        existing = 0
        for char in reversed(''.join(self.output)):
            if char == '\n':
                existing += 1
            elif char == ' ':
                continue
            else:
                break

        # Add only the needed newlines
        needed = max(0, count - existing)
        if needed > 0:
            self.output.append('\n' * needed)

    def get_markdown(self) -> str:
        """Get the converted Markdown text."""
        text = ''.join(self.output)
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+\n', '\n', text)
        return text.strip()


def html_to_markdown(html: str) -> str:
    """Convert HTML to clean Markdown.

    Extracts content from main/article/body, converts to Markdown format,
    and strips navigation, sidebars, and other non-content elements.

    Args:
        html: HTML content to convert.

    Returns:
        Clean Markdown string.
    """
    converter = HTMLToMarkdownConverter()
    try:
        converter.feed(html)
    except Exception:
        # On parse error, return empty string
        return ''
    return converter.get_markdown()


def extract_markdown_headings(markdown: str) -> List[dict]:
    """Extract headings from Markdown text.

    Args:
        markdown: Markdown text.

    Returns:
        List of heading dicts with 'level', 'text', and 'position' keys.
    """
    headings = []
    # Match markdown headings: # Heading, ## Heading, etc.
    pattern = r'^(#{1,6})\s+(.+?)(?:\s*#*)?$'

    for match in re.finditer(pattern, markdown, re.MULTILINE):
        level = len(match.group(1))
        text = match.group(2).strip()
        headings.append({
            'level': level,
            'text': text,
            'position': match.start()
        })

    return headings


def split_markdown_by_headings(markdown: str) -> List[dict]:
    """Split Markdown text into sections based on headings.

    Args:
        markdown: Markdown text.

    Returns:
        List of section dicts with 'content', 'heading', 'level', and 'heading_path'.
    """
    headings = extract_markdown_headings(markdown)

    if not headings:
        return [{'content': markdown, 'heading': '', 'level': 0, 'heading_path': ''}]

    sections = []
    heading_stack = {}  # level -> heading text

    for i, heading in enumerate(headings):
        start = heading['position']
        end = headings[i + 1]['position'] if i + 1 < len(headings) else len(markdown)

        content = markdown[start:end].strip()
        level = heading['level']
        text = heading['text']

        # Update heading stack - clear lower levels when we see a higher level
        for l in list(heading_stack.keys()):
            if l >= level:
                del heading_stack[l]
        heading_stack[level] = text

        # Build heading path from stack
        path_parts = [heading_stack[l] for l in sorted(heading_stack.keys())]
        heading_path = ' > '.join(path_parts)

        sections.append({
            'content': content,
            'heading': text,
            'level': level,
            'heading_path': heading_path
        })

    return sections
