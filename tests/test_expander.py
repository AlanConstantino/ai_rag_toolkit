"""Tests for the query expander module."""

import unittest
from unittest.mock import MagicMock


class TestQueryExpander(unittest.TestCase):
    """Test query expansion."""

    def test_expand_adds_variations(self):
        """expand should add query variations."""
        from rag_system.query.expander import QueryExpander

        expander = QueryExpander()
        expanded = expander.expand("How to install Redis")

        self.assertIn("How to install Redis", expanded)
        self.assertGreater(len(expanded), 1)

    def test_expand_includes_original(self):
        """expand should always include original query."""
        from rag_system.query.expander import QueryExpander

        expander = QueryExpander()
        original = "Redis configuration"
        expanded = expander.expand(original)

        self.assertEqual(expanded[0], original)

    def test_expand_with_synonyms(self):
        """expand should add synonym variations."""
        from rag_system.query.expander import QueryExpander

        expander = QueryExpander()
        expanded = expander.expand("How to configure settings")

        # Should include variations with "setup" or "set up"
        variations = ' '.join(expanded).lower()
        self.assertTrue('configure' in variations or 'setup' in variations)

    def test_expand_handles_empty_query(self):
        """expand should handle empty query."""
        from rag_system.query.expander import QueryExpander

        expander = QueryExpander()
        expanded = expander.expand("")

        self.assertEqual(expanded, [""])


class TestQueryExpanderWithLLM(unittest.TestCase):
    """Test query expansion with LLM."""

    def test_expand_with_llm(self):
        """expand_with_llm should use LLM for expansion."""
        from rag_system.query.expander import QueryExpander

        mock_client = MagicMock()
        mock_client.complete.return_value = "Redis installation\nInstalling Redis\nRedis setup"

        expander = QueryExpander(chat_client=mock_client)
        expanded = expander.expand_with_llm("How to install Redis")

        self.assertIn("How to install Redis", expanded)
        self.assertIn("Redis installation", expanded)
        mock_client.complete.assert_called_once()

    def test_expand_with_llm_handles_error(self):
        """expand_with_llm should fall back to rule-based on error."""
        from rag_system.query.expander import QueryExpander

        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("API error")

        expander = QueryExpander(chat_client=mock_client)
        expanded = expander.expand_with_llm("How to install Redis")

        # Should fall back to rule-based expansion
        self.assertIn("How to install Redis", expanded)


class TestSynonymExpansion(unittest.TestCase):
    """Test synonym-based expansion."""

    def test_get_synonyms(self):
        """get_synonyms should return known synonyms."""
        from rag_system.query.expander import get_synonyms

        self.assertIn('setup', get_synonyms('configure'))
        self.assertIn('install', get_synonyms('set up'))

    def test_get_synonyms_unknown_word(self):
        """get_synonyms should return empty for unknown words."""
        from rag_system.query.expander import get_synonyms

        self.assertEqual(get_synonyms('xyzabc'), [])

    def test_expand_with_synonyms(self):
        """expand_with_synonyms should create variations."""
        from rag_system.query.expander import expand_with_synonyms

        variations = expand_with_synonyms("how to configure")
        self.assertIn("how to configure", variations)
        self.assertTrue(len(variations) > 1)


class TestPhraseExtraction(unittest.TestCase):
    """Test key phrase extraction."""

    def test_extract_key_terms(self):
        """extract_key_terms should find important terms."""
        from rag_system.query.expander import extract_key_terms

        terms = extract_key_terms("How do I configure Redis caching?")
        self.assertIn('redis', terms)
        self.assertIn('caching', terms)
        self.assertIn('configure', terms)

    def test_extract_key_terms_removes_stopwords(self):
        """extract_key_terms should remove common words."""
        from rag_system.query.expander import extract_key_terms

        terms = extract_key_terms("What is the best way to do this?")
        self.assertNotIn('is', terms)
        self.assertNotIn('the', terms)
        self.assertNotIn('to', terms)


if __name__ == '__main__':
    unittest.main()
