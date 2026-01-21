"""Tests for token-aware chunking functionality.

Tests the ability to chunk text based on token counts rather than
character counts for more accurate LLM context window management.
"""

import unittest

from rag_system.ingestion.chunker import (
    estimate_token_count,
    chunk_text,
    chunk_markdown
)
from rag_system import config


class TestTokenEstimation(unittest.TestCase):
    """Tests for token count estimation."""

    def test_empty_string(self):
        """estimate_token_count should return 0 for empty string."""
        self.assertEqual(estimate_token_count(''), 0)

    def test_single_word(self):
        """estimate_token_count should count single words."""
        count = estimate_token_count('hello')
        self.assertGreaterEqual(count, 1)

    def test_simple_sentence(self):
        """estimate_token_count should handle simple sentences."""
        text = 'The quick brown fox jumps over the lazy dog.'
        count = estimate_token_count(text)
        # Should be roughly 9-11 tokens (9 words + punctuation)
        self.assertGreater(count, 5)
        self.assertLess(count, 20)

    def test_code_snippet(self):
        """estimate_token_count should handle code with special chars."""
        code = 'def hello_world():\n    print("Hello!")'
        count = estimate_token_count(code)
        # Code typically tokenizes into more tokens than words
        self.assertGreater(count, 3)

    def test_whitespace_only(self):
        """estimate_token_count should handle whitespace-only text."""
        self.assertEqual(estimate_token_count('   \n\t  '), 0)

    def test_punctuation_creates_tokens(self):
        """estimate_token_count should count punctuation as potential tokens."""
        # "Hello, world!" should count punctuation
        with_punct = estimate_token_count('Hello, world!')
        without_punct = estimate_token_count('Hello world')
        # With punctuation should have same or more tokens
        self.assertGreaterEqual(with_punct, without_punct)


class TestChunkTextWithTokens(unittest.TestCase):
    """Tests for chunk_text with token-based sizing."""

    def test_chunk_by_tokens(self):
        """chunk_text should respect token limits when use_tokens=True."""
        # Create text with known token count (roughly 100 words = ~100 tokens)
        words = ['word'] * 100
        text = ' '.join(words)

        chunks = chunk_text(text, chunk_size=25, overlap=5, use_tokens=True)

        # Should create multiple chunks
        self.assertGreater(len(chunks), 1)

        # Each chunk should be roughly within the token limit
        for chunk in chunks:
            token_count = estimate_token_count(chunk)
            # Allow some tolerance since we break at sentence/word boundaries
            self.assertLess(token_count, 40)  # 25 + buffer for boundary seeking

    def test_chunk_by_characters_default(self):
        """chunk_text should use character sizing by default."""
        text = 'a' * 1000
        chunks = chunk_text(text, chunk_size=300, overlap=50)

        # With character mode, 1000 chars / 300 chunk size = ~4 chunks
        self.assertGreater(len(chunks), 2)

    def test_small_text_single_chunk_tokens(self):
        """chunk_text should return single chunk for small text with tokens."""
        text = 'Hello world'
        chunks = chunk_text(text, chunk_size=100, overlap=10, use_tokens=True)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], 'Hello world')


class TestChunkMarkdownWithTokens(unittest.TestCase):
    """Tests for chunk_markdown with token awareness."""

    def test_token_aware_config_exists(self):
        """Config should have USE_TOKEN_CHUNKING setting."""
        self.assertTrue(hasattr(config, 'USE_TOKEN_CHUNKING'))

    def test_small_chunk_tokens_config(self):
        """Config should have SMALL_CHUNK_TOKENS setting."""
        self.assertTrue(hasattr(config, 'SMALL_CHUNK_TOKENS'))
        self.assertIsInstance(config.SMALL_CHUNK_TOKENS, int)
        self.assertGreater(config.SMALL_CHUNK_TOKENS, 0)

    def test_large_chunk_tokens_config(self):
        """Config should have LARGE_CHUNK_TOKENS setting."""
        self.assertTrue(hasattr(config, 'LARGE_CHUNK_TOKENS'))
        self.assertIsInstance(config.LARGE_CHUNK_TOKENS, int)
        self.assertGreater(config.LARGE_CHUNK_TOKENS, config.SMALL_CHUNK_TOKENS)


class TestTokenCharacterRatio(unittest.TestCase):
    """Tests for token to character ratio estimation."""

    def test_english_text_ratio(self):
        """English text should have roughly 4 chars per token."""
        # Average English text is about 4 characters per token
        text = 'The quick brown fox jumps over the lazy dog.'
        tokens = estimate_token_count(text)
        chars = len(text)

        # Should be roughly 4 chars per token (allow 2-6 range)
        ratio = chars / tokens if tokens > 0 else 0
        self.assertGreater(ratio, 2)
        self.assertLess(ratio, 8)

    def test_consistent_estimation(self):
        """Token estimation should be consistent for same input."""
        text = 'This is a test sentence for consistency checking.'
        count1 = estimate_token_count(text)
        count2 = estimate_token_count(text)
        self.assertEqual(count1, count2)


if __name__ == '__main__':
    unittest.main()
