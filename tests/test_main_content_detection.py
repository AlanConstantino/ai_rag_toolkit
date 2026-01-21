"""Tests for main content detection functionality.

Tests the ability to identify and extract the main content area of HTML pages,
filtering out navigation, sidebars, footers, and other boilerplate.
"""

import unittest

from rag_system.ingestion.parser import (
    extract_main_content,
    find_main_content_element
)


class TestMainContentDetection(unittest.TestCase):
    """Tests for main content detection."""

    def test_extracts_main_tag_content(self):
        """extract_main_content should prioritize <main> tag."""
        html = '''
        <html>
        <body>
            <nav>Navigation links here</nav>
            <main>
                <h1>Main Article Title</h1>
                <p>This is the main content of the page.</p>
            </main>
            <footer>Footer content</footer>
        </body>
        </html>
        '''
        content = extract_main_content(html)

        self.assertIn('Main Article Title', content)
        self.assertIn('main content of the page', content)
        self.assertNotIn('Navigation links', content)
        self.assertNotIn('Footer content', content)

    def test_extracts_article_tag_content(self):
        """extract_main_content should use <article> tag when no <main>."""
        html = '''
        <html>
        <body>
            <nav>Navigation</nav>
            <article>
                <h1>Article Title</h1>
                <p>Article body content here.</p>
            </article>
            <aside>Sidebar content</aside>
        </body>
        </html>
        '''
        content = extract_main_content(html)

        self.assertIn('Article Title', content)
        self.assertIn('Article body content', content)
        self.assertNotIn('Navigation', content)
        self.assertNotIn('Sidebar content', content)

    def test_extracts_role_main_content(self):
        """extract_main_content should use elements with role='main'."""
        html = '''
        <html>
        <body>
            <header>Header content</header>
            <div role="main">
                <h1>Main Content</h1>
                <p>This is the primary content area.</p>
            </div>
            <footer>Footer</footer>
        </body>
        </html>
        '''
        content = extract_main_content(html)

        self.assertIn('Main Content', content)
        self.assertIn('primary content area', content)
        self.assertNotIn('Header content', content)
        self.assertNotIn('Footer', content)

    def test_extracts_content_class_patterns(self):
        """extract_main_content should recognize common content class names."""
        html = '''
        <html>
        <body>
            <div class="sidebar">Sidebar links</div>
            <div class="main-content">
                <h1>Page Title</h1>
                <p>The actual content is here.</p>
            </div>
            <div class="footer-links">More links</div>
        </body>
        </html>
        '''
        content = extract_main_content(html)

        self.assertIn('Page Title', content)
        self.assertIn('actual content', content)
        self.assertNotIn('Sidebar links', content)

    def test_extracts_content_id_patterns(self):
        """extract_main_content should recognize common content id names."""
        html = '''
        <html>
        <body>
            <div id="navigation">Nav</div>
            <div id="content">
                <h1>Content Title</h1>
                <p>Main body text.</p>
            </div>
        </body>
        </html>
        '''
        content = extract_main_content(html)

        self.assertIn('Content Title', content)
        self.assertIn('Main body text', content)

    def test_fallback_to_full_content(self):
        """extract_main_content should fall back to full content if no main detected."""
        html = '''
        <html>
        <body>
            <div>
                <h1>Simple Page</h1>
                <p>Content without semantic structure.</p>
            </div>
        </body>
        </html>
        '''
        content = extract_main_content(html)

        self.assertIn('Simple Page', content)
        self.assertIn('without semantic structure', content)

    def test_handles_nested_main_content(self):
        """extract_main_content should handle nested content areas."""
        html = '''
        <html>
        <body>
            <nav>Nav</nav>
            <main>
                <article>
                    <h1>Nested Article</h1>
                    <p>Article content within main.</p>
                </article>
            </main>
        </body>
        </html>
        '''
        content = extract_main_content(html)

        self.assertIn('Nested Article', content)
        self.assertIn('Article content', content)
        self.assertNotIn('Nav', content)

    def test_excludes_navigation_elements(self):
        """extract_main_content should exclude nav, header, footer, aside."""
        html = '''
        <html>
        <body>
            <header>Site Header</header>
            <nav>Main Navigation</nav>
            <aside>Related Links</aside>
            <main>
                <p>Main content paragraph.</p>
            </main>
            <footer>Site Footer</footer>
        </body>
        </html>
        '''
        content = extract_main_content(html)

        self.assertIn('Main content paragraph', content)
        self.assertNotIn('Site Header', content)
        self.assertNotIn('Main Navigation', content)
        self.assertNotIn('Related Links', content)
        self.assertNotIn('Site Footer', content)


class TestFindMainContentElement(unittest.TestCase):
    """Tests for finding the main content element."""

    def test_finds_main_tag(self):
        """find_main_content_element should return 'main' for <main> tag."""
        html = '<html><body><main><p>Content</p></main></body></html>'
        element = find_main_content_element(html)
        self.assertEqual(element, 'main')

    def test_finds_article_tag(self):
        """find_main_content_element should return 'article' for <article> tag."""
        html = '<html><body><article><p>Content</p></article></body></html>'
        element = find_main_content_element(html)
        self.assertEqual(element, 'article')

    def test_finds_role_main(self):
        """find_main_content_element should return 'role=main' for role attribute."""
        html = '<html><body><div role="main"><p>Content</p></div></body></html>'
        element = find_main_content_element(html)
        self.assertEqual(element, 'role=main')

    def test_finds_content_class(self):
        """find_main_content_element should detect content class patterns."""
        html = '<html><body><div class="main-content"><p>Content</p></div></body></html>'
        element = find_main_content_element(html)
        self.assertIn('class=', element)

    def test_returns_none_for_no_main(self):
        """find_main_content_element should return None if no main detected."""
        html = '<html><body><div><p>Content</p></div></body></html>'
        element = find_main_content_element(html)
        self.assertIsNone(element)


class TestMainContentConfig(unittest.TestCase):
    """Tests for main content detection configuration."""

    def test_main_content_patterns_configurable(self):
        """Config should have MAIN_CONTENT_SELECTORS setting."""
        from rag_system import config
        self.assertTrue(hasattr(config, 'MAIN_CONTENT_SELECTORS'))
        self.assertIsInstance(config.MAIN_CONTENT_SELECTORS, (list, tuple))


if __name__ == '__main__':
    unittest.main()
