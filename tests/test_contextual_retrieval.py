"""Tests for contextual retrieval functionality.

Tests the contextual embedding feature where chunks are embedded with
surrounding context (page title, heading path) for better retrieval.
"""

import unittest

from rag_system.ingestion.indexer import build_contextual_text


class TestBuildContextualText(unittest.TestCase):
    """Tests for build_contextual_text function."""

    def test_build_with_all_context(self):
        """build_contextual_text should include all context parts."""
        chunk = {
            'content': 'This is the chunk content.',
            'heading_path': 'Configuration > Database'
        }
        result = build_contextual_text(chunk, page_title='API Reference')

        self.assertIn('Document: API Reference', result)
        self.assertIn('Section: Configuration > Database', result)
        self.assertIn('This is the chunk content.', result)

    def test_build_without_heading_path(self):
        """build_contextual_text should work without heading path."""
        chunk = {
            'content': 'This is the chunk content.',
            'heading_path': ''
        }
        result = build_contextual_text(chunk, page_title='API Reference')

        self.assertIn('Document: API Reference', result)
        self.assertNotIn('Section:', result)
        self.assertIn('This is the chunk content.', result)

    def test_build_without_page_title(self):
        """build_contextual_text should work without page title."""
        chunk = {
            'content': 'This is the chunk content.',
            'heading_path': 'Configuration > Database'
        }
        result = build_contextual_text(chunk, page_title=None)

        self.assertNotIn('Document:', result)
        self.assertIn('Section: Configuration > Database', result)
        self.assertIn('This is the chunk content.', result)

    def test_build_with_only_content(self):
        """build_contextual_text should return content when no context."""
        chunk = {
            'content': 'This is the chunk content.',
            'heading_path': ''
        }
        result = build_contextual_text(chunk, page_title=None)

        self.assertEqual(result.strip(), 'This is the chunk content.')

    def test_build_preserves_content_formatting(self):
        """build_contextual_text should preserve content formatting."""
        chunk = {
            'content': '# Heading\n\nParagraph with **bold** text.',
            'heading_path': 'API'
        }
        result = build_contextual_text(chunk, page_title='Docs')

        self.assertIn('# Heading\n\nParagraph with **bold** text.', result)

    def test_build_handles_long_heading_path(self):
        """build_contextual_text should handle deep heading hierarchies."""
        chunk = {
            'content': 'Content here.',
            'heading_path': 'API > Authentication > OAuth2 > Token Refresh'
        }
        result = build_contextual_text(chunk, page_title='Developer Guide')

        self.assertIn('Section: API > Authentication > OAuth2 > Token Refresh', result)


class TestContextualRetrievalIntegration(unittest.TestCase):
    """Integration tests for contextual retrieval in indexer."""

    def test_chunks_contain_contextual_embedding_text(self):
        """Indexer should build contextual text for embedding."""
        chunk = {
            'content': 'Configure the database connection string.',
            'heading_path': 'Configuration > Database > Connection'
        }

        contextual = build_contextual_text(chunk, page_title='Setup Guide')

        self.assertIn('Setup Guide', contextual)
        self.assertIn('Configuration > Database > Connection', contextual)
        self.assertIn('Configure the database connection string.', contextual)

        lines = contextual.strip().split('\n')
        self.assertTrue(any('Document:' in line for line in lines))
        self.assertTrue(any('Section:' in line for line in lines))


if __name__ == '__main__':
    unittest.main()
