"""Tests for the parser module."""

import unittest


class TestHTMLParser(unittest.TestCase):
    """Test HTML parsing functionality."""

    def test_extract_title(self):
        """extract_title should get the page title."""
        from rag_system.ingestion.parser import extract_title

        html = '<html><head><title>My Page Title</title></head><body></body></html>'

        title = extract_title(html)

        self.assertEqual(title, 'My Page Title')

    def test_extract_title_missing(self):
        """extract_title should return empty string for missing title."""
        from rag_system.ingestion.parser import extract_title

        html = '<html><head></head><body>No title</body></html>'

        title = extract_title(html)

        self.assertEqual(title, '')

    def test_extract_text_basic(self):
        """extract_text should get text content from HTML."""
        from rag_system.ingestion.parser import extract_text

        html = '<html><body><p>Hello world</p></body></html>'

        text = extract_text(html)

        self.assertIn('Hello world', text)

    def test_extract_text_removes_scripts(self):
        """extract_text should remove script content."""
        from rag_system.ingestion.parser import extract_text

        html = '''
        <html><body>
            <p>Visible text</p>
            <script>var x = "invisible";</script>
        </body></html>
        '''

        text = extract_text(html)

        self.assertIn('Visible text', text)
        self.assertNotIn('invisible', text)
        self.assertNotIn('var x', text)

    def test_extract_text_removes_styles(self):
        """extract_text should remove style content."""
        from rag_system.ingestion.parser import extract_text

        html = '''
        <html><body>
            <p>Visible text</p>
            <style>.hidden { display: none; }</style>
        </body></html>
        '''

        text = extract_text(html)

        self.assertIn('Visible text', text)
        self.assertNotIn('display: none', text)

    def test_extract_text_removes_nav(self):
        """extract_text should optionally remove navigation."""
        from rag_system.ingestion.parser import extract_text

        html = '''
        <html><body>
            <nav><a href="/">Home</a></nav>
            <main><p>Main content</p></main>
        </body></html>
        '''

        text = extract_text(html, remove_nav=True)

        self.assertIn('Main content', text)
        self.assertNotIn('Home', text)

    def test_extract_text_removes_footer(self):
        """extract_text should optionally remove footer."""
        from rag_system.ingestion.parser import extract_text

        html = '''
        <html><body>
            <article><p>Article content</p></article>
            <footer>Copyright 2024</footer>
        </body></html>
        '''

        text = extract_text(html, remove_footer=True)

        self.assertIn('Article content', text)
        self.assertNotIn('Copyright', text)

    def test_extract_text_preserves_headings(self):
        """extract_text should preserve heading structure."""
        from rag_system.ingestion.parser import extract_text

        html = '''
        <html><body>
            <h1>Main Title</h1>
            <p>Introduction</p>
            <h2>Section 1</h2>
            <p>Section content</p>
        </body></html>
        '''

        text = extract_text(html)

        # Check order is preserved
        main_pos = text.find('Main Title')
        intro_pos = text.find('Introduction')
        section_pos = text.find('Section 1')

        self.assertLess(main_pos, intro_pos)
        self.assertLess(intro_pos, section_pos)

    def test_extract_text_handles_lists(self):
        """extract_text should handle lists properly."""
        from rag_system.ingestion.parser import extract_text

        html = '''
        <html><body>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </body></html>
        '''

        text = extract_text(html)

        self.assertIn('Item 1', text)
        self.assertIn('Item 2', text)

    def test_extract_text_handles_tables(self):
        """extract_text should extract table content."""
        from rag_system.ingestion.parser import extract_text

        html = '''
        <html><body>
            <table>
                <tr><th>Header</th></tr>
                <tr><td>Cell value</td></tr>
            </table>
        </body></html>
        '''

        text = extract_text(html)

        self.assertIn('Header', text)
        self.assertIn('Cell value', text)

    def test_extract_text_normalizes_whitespace(self):
        """extract_text should normalize whitespace."""
        from rag_system.ingestion.parser import extract_text

        html = '<html><body><p>Text   with    extra   spaces</p></body></html>'

        text = extract_text(html)

        self.assertNotIn('   ', text)  # No triple spaces

    def test_extract_text_handles_code_blocks(self):
        """extract_text should preserve code block content."""
        from rag_system.ingestion.parser import extract_text

        html = '''
        <html><body>
            <pre><code>def hello():
    print("Hello")</code></pre>
        </body></html>
        '''

        text = extract_text(html)

        self.assertIn('def hello', text)
        self.assertIn('print', text)


class TestHeadingExtraction(unittest.TestCase):
    """Test heading extraction functionality."""

    def test_extract_headings(self):
        """extract_headings should return heading hierarchy."""
        from rag_system.ingestion.parser import extract_headings

        html = '''
        <html><body>
            <h1>Main Title</h1>
            <h2>Section A</h2>
            <h3>Subsection A1</h3>
            <h2>Section B</h2>
        </body></html>
        '''

        headings = extract_headings(html)

        self.assertEqual(len(headings), 4)
        self.assertEqual(headings[0], {'level': 1, 'text': 'Main Title'})
        self.assertEqual(headings[1], {'level': 2, 'text': 'Section A'})

    def test_extract_headings_empty(self):
        """extract_headings should return empty list for no headings."""
        from rag_system.ingestion.parser import extract_headings

        html = '<html><body><p>No headings here</p></body></html>'

        headings = extract_headings(html)

        self.assertEqual(headings, [])


class TestMetadataExtraction(unittest.TestCase):
    """Test metadata extraction functionality."""

    def test_extract_meta_description(self):
        """extract_metadata should get meta description."""
        from rag_system.ingestion.parser import extract_metadata

        html = '''
        <html>
        <head>
            <meta name="description" content="Page description here">
        </head>
        <body></body>
        </html>
        '''

        metadata = extract_metadata(html)

        self.assertEqual(metadata.get('description'), 'Page description here')

    def test_extract_meta_keywords(self):
        """extract_metadata should get meta keywords."""
        from rag_system.ingestion.parser import extract_metadata

        html = '''
        <html>
        <head>
            <meta name="keywords" content="python, programming, tutorial">
        </head>
        <body></body>
        </html>
        '''

        metadata = extract_metadata(html)

        self.assertEqual(metadata.get('keywords'), 'python, programming, tutorial')

    def test_extract_og_title(self):
        """extract_metadata should get OpenGraph title."""
        from rag_system.ingestion.parser import extract_metadata

        html = '''
        <html>
        <head>
            <meta property="og:title" content="OG Title">
        </head>
        <body></body>
        </html>
        '''

        metadata = extract_metadata(html)

        self.assertEqual(metadata.get('og:title'), 'OG Title')


class TestParseHTML(unittest.TestCase):
    """Test the main parse_html function."""

    def test_parse_html_returns_all_fields(self):
        """parse_html should return complete parsed result."""
        from rag_system.ingestion.parser import parse_html

        html = '''
        <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="Test description">
        </head>
        <body>
            <h1>Welcome</h1>
            <p>Content here</p>
        </body>
        </html>
        '''

        result = parse_html(html)

        self.assertIn('title', result)
        self.assertIn('text', result)
        self.assertIn('headings', result)
        self.assertIn('metadata', result)

        self.assertEqual(result['title'], 'Test Page')
        self.assertIn('Welcome', result['text'])
        self.assertIn('Content here', result['text'])

    def test_parse_html_handles_malformed(self):
        """parse_html should handle malformed HTML gracefully."""
        from rag_system.ingestion.parser import parse_html

        html = '<html><body><p>Unclosed paragraph<div>Mixed</body>'

        # Should not raise
        result = parse_html(html)

        self.assertIn('text', result)


if __name__ == '__main__':
    unittest.main()
