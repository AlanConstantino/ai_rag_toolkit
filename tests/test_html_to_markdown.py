"""Tests for the html_to_markdown module."""

import unittest


class TestHTMLToMarkdownConverter(unittest.TestCase):
    """Test HTML to Markdown conversion."""

    def test_basic_paragraph(self):
        """html_to_markdown should convert paragraphs."""
        from rag_system.ingestion.html_to_markdown import html_to_markdown

        html = '<html><body><p>Hello world</p></body></html>'
        result = html_to_markdown(html)

        self.assertIn('Hello world', result)

    def test_headings_conversion(self):
        """html_to_markdown should convert headings to markdown format."""
        from rag_system.ingestion.html_to_markdown import html_to_markdown

        html = '''
        <html><body>
            <h1>Main Title</h1>
            <h2>Section</h2>
            <h3>Subsection</h3>
        </body></html>
        '''
        result = html_to_markdown(html)

        self.assertIn('# Main Title', result)
        self.assertIn('## Section', result)
        self.assertIn('### Subsection', result)

    def test_skips_nav_elements(self):
        """html_to_markdown should skip content in nav elements."""
        from rag_system.ingestion.html_to_markdown import html_to_markdown

        html = '''
        <html><body>
            <nav><h3>Navigation</h3><a href="/">Home</a></nav>
            <main><h1>Real Content</h1><p>This is the content.</p></main>
        </body></html>
        '''
        result = html_to_markdown(html)

        self.assertNotIn('Navigation', result)
        self.assertNotIn('Home', result)
        self.assertIn('# Real Content', result)
        self.assertIn('This is the content', result)

    def test_skips_aside_elements(self):
        """html_to_markdown should skip content in aside elements."""
        from rag_system.ingestion.html_to_markdown import html_to_markdown

        html = '''
        <html><body>
            <aside><h4>Sidebar</h4><p>Side content</p></aside>
            <article><h1>Article</h1><p>Main content</p></article>
        </body></html>
        '''
        result = html_to_markdown(html)

        self.assertNotIn('Sidebar', result)
        self.assertNotIn('Side content', result)
        self.assertIn('# Article', result)
        self.assertIn('Main content', result)

    def test_skips_footer_elements(self):
        """html_to_markdown should skip content in footer elements."""
        from rag_system.ingestion.html_to_markdown import html_to_markdown

        html = '''
        <html><body>
            <main><h1>Content</h1><p>Main text</p></main>
            <footer><p>Copyright 2024</p></footer>
        </body></html>
        '''
        result = html_to_markdown(html)

        self.assertIn('# Content', result)
        self.assertIn('Main text', result)
        self.assertNotIn('Copyright', result)

    def test_skips_header_elements(self):
        """html_to_markdown should skip content in header elements."""
        from rag_system.ingestion.html_to_markdown import html_to_markdown

        html = '''
        <html><body>
            <header><h1>Site Title</h1></header>
            <main><h1>Page Title</h1><p>Content here</p></main>
        </body></html>
        '''
        result = html_to_markdown(html)

        self.assertNotIn('Site Title', result)
        self.assertIn('# Page Title', result)
        self.assertIn('Content here', result)

    def test_bold_and_italic(self):
        """html_to_markdown should convert bold and italic."""
        from rag_system.ingestion.html_to_markdown import html_to_markdown

        html = '<p>This is <strong>bold</strong> and <em>italic</em>.</p>'
        result = html_to_markdown(html)

        self.assertIn('**bold**', result)
        self.assertIn('*italic*', result)

    def test_inline_code(self):
        """html_to_markdown should convert inline code."""
        from rag_system.ingestion.html_to_markdown import html_to_markdown

        html = '<p>Use <code>print()</code> to output.</p>'
        result = html_to_markdown(html)

        self.assertIn('`print()`', result)

    def test_code_blocks(self):
        """html_to_markdown should convert code blocks."""
        from rag_system.ingestion.html_to_markdown import html_to_markdown

        html = '<pre><code>def hello():\n    print("Hi")</code></pre>'
        result = html_to_markdown(html)

        self.assertIn('```', result)
        self.assertIn('def hello():', result)

    def test_unordered_lists(self):
        """html_to_markdown should convert unordered lists."""
        from rag_system.ingestion.html_to_markdown import html_to_markdown

        html = '<ul><li>Item 1</li><li>Item 2</li></ul>'
        result = html_to_markdown(html)

        self.assertIn('- Item 1', result)
        self.assertIn('- Item 2', result)

    def test_ordered_lists(self):
        """html_to_markdown should convert ordered lists."""
        from rag_system.ingestion.html_to_markdown import html_to_markdown

        html = '<ol><li>First</li><li>Second</li></ol>'
        result = html_to_markdown(html)

        self.assertIn('1. First', result)
        self.assertIn('2. Second', result)

    def test_links(self):
        """html_to_markdown should convert links."""
        from rag_system.ingestion.html_to_markdown import html_to_markdown

        html = '<p>Visit <a href="https://example.com">Example</a>.</p>'
        result = html_to_markdown(html)

        self.assertIn('[Example](https://example.com)', result)

    def test_skips_script_and_style(self):
        """html_to_markdown should skip script and style elements."""
        from rag_system.ingestion.html_to_markdown import html_to_markdown

        html = '''
        <html>
        <head><style>.hidden { display: none; }</style></head>
        <body>
            <script>alert("hi");</script>
            <p>Visible content</p>
        </body></html>
        '''
        result = html_to_markdown(html)

        self.assertNotIn('display: none', result)
        self.assertNotIn('alert', result)
        self.assertIn('Visible content', result)

    def test_empty_html(self):
        """html_to_markdown should handle empty HTML."""
        from rag_system.ingestion.html_to_markdown import html_to_markdown

        result = html_to_markdown('')

        self.assertEqual(result, '')


class TestExtractMarkdownHeadings(unittest.TestCase):
    """Test Markdown heading extraction."""

    def test_extracts_all_heading_levels(self):
        """extract_markdown_headings should find all heading levels."""
        from rag_system.ingestion.html_to_markdown import extract_markdown_headings

        markdown = '''# H1
## H2
### H3
#### H4
##### H5
###### H6
'''
        headings = extract_markdown_headings(markdown)

        self.assertEqual(len(headings), 6)
        self.assertEqual(headings[0]['level'], 1)
        self.assertEqual(headings[0]['text'], 'H1')
        self.assertEqual(headings[5]['level'], 6)
        self.assertEqual(headings[5]['text'], 'H6')

    def test_includes_position(self):
        """extract_markdown_headings should include heading position."""
        from rag_system.ingestion.html_to_markdown import extract_markdown_headings

        markdown = '''Some intro text.

# First Heading

Content here.

## Second Heading
'''
        headings = extract_markdown_headings(markdown)

        self.assertEqual(len(headings), 2)
        self.assertGreater(headings[0]['position'], 0)
        self.assertGreater(headings[1]['position'], headings[0]['position'])

    def test_no_headings(self):
        """extract_markdown_headings should handle text with no headings."""
        from rag_system.ingestion.html_to_markdown import extract_markdown_headings

        markdown = 'Just plain text with no headings.'
        headings = extract_markdown_headings(markdown)

        self.assertEqual(len(headings), 0)


class TestSplitMarkdownByHeadings(unittest.TestCase):
    """Test Markdown section splitting."""

    def test_splits_by_headings(self):
        """split_markdown_by_headings should split at heading boundaries."""
        from rag_system.ingestion.html_to_markdown import split_markdown_by_headings

        markdown = '''# Introduction

This is the intro.

## Getting Started

Start here.

## Advanced Topics

More complex stuff.
'''
        sections = split_markdown_by_headings(markdown)

        self.assertEqual(len(sections), 3)
        self.assertEqual(sections[0]['heading'], 'Introduction')
        self.assertEqual(sections[1]['heading'], 'Getting Started')
        self.assertEqual(sections[2]['heading'], 'Advanced Topics')

    def test_builds_heading_paths(self):
        """split_markdown_by_headings should build hierarchical heading paths."""
        from rag_system.ingestion.html_to_markdown import split_markdown_by_headings

        markdown = '''# Main

## Section A

### Subsection A1

## Section B
'''
        sections = split_markdown_by_headings(markdown)

        self.assertEqual(sections[0]['heading_path'], 'Main')
        self.assertEqual(sections[1]['heading_path'], 'Main > Section A')
        self.assertEqual(sections[2]['heading_path'], 'Main > Section A > Subsection A1')
        self.assertEqual(sections[3]['heading_path'], 'Main > Section B')

    def test_no_headings_returns_single_section(self):
        """split_markdown_by_headings should return one section for no headings."""
        from rag_system.ingestion.html_to_markdown import split_markdown_by_headings

        markdown = 'Just plain text with no headings.'
        sections = split_markdown_by_headings(markdown)

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]['content'], markdown)
        self.assertEqual(sections[0]['heading_path'], '')


class TestMarkdownChunking(unittest.TestCase):
    """Test the full Markdown chunking pipeline."""

    def test_chunk_markdown_creates_chunks(self):
        """chunk_markdown should create large and small chunks."""
        from rag_system.ingestion.chunker import chunk_markdown

        markdown = '''# Introduction

This is a fairly long introduction that goes on and on with lots of text
to ensure we get multiple chunks out of it. We need enough content here
to exceed the chunk size limits. Let me add more text to make this work.
More and more text keeps flowing to build up the content size.

## Getting Started

Here we have another section with plenty of content to chunk.
This section also needs to be long enough to create multiple chunks.
Adding more sentences to bulk up the content for testing purposes.
'''
        result = chunk_markdown(markdown, small_chunk_size=100, large_chunk_size=200, overlap=20)

        self.assertIn('large_chunks', result)
        self.assertIn('small_chunks', result)
        self.assertGreater(len(result['large_chunks']), 0)
        self.assertGreater(len(result['small_chunks']), 0)

    def test_chunk_markdown_preserves_heading_paths(self):
        """chunk_markdown should assign correct heading paths to chunks."""
        from rag_system.ingestion.chunker import chunk_markdown

        markdown = '''# Main Title

Content for main.

## Section One

Content for section one with enough text to create a chunk.
'''
        result = chunk_markdown(markdown, small_chunk_size=50, large_chunk_size=100, overlap=10)

        # Check that chunks have heading paths
        for chunk in result['large_chunks']:
            self.assertIn('heading_path', chunk)

        # First section should have "Main Title" in path
        paths = [c['heading_path'] for c in result['large_chunks']]
        self.assertTrue(any('Main Title' in p for p in paths))

    def test_chunk_markdown_empty_input(self):
        """chunk_markdown should handle empty input."""
        from rag_system.ingestion.chunker import chunk_markdown

        result = chunk_markdown('')

        self.assertEqual(result['large_chunks'], [])
        self.assertEqual(result['small_chunks'], [])


if __name__ == '__main__':
    unittest.main()
