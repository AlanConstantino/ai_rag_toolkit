"""Tests for the query classifier module."""

import unittest
from unittest.mock import MagicMock


class TestQueryClassifier(unittest.TestCase):
    """Test query classification."""

    def test_classify_factual_query(self):
        """classify should identify factual queries."""
        from rag_system.query.classifier import QueryClassifier

        classifier = QueryClassifier()

        # Factual questions about specific things
        self.assertEqual(classifier.classify("What is Redis?"), "factual")
        self.assertEqual(classifier.classify("What does the config option do?"), "factual")

    def test_classify_procedural_query(self):
        """classify should identify procedural queries."""
        from rag_system.query.classifier import QueryClassifier

        classifier = QueryClassifier()

        # How-to questions
        self.assertEqual(classifier.classify("How do I install Redis?"), "procedural")
        self.assertEqual(classifier.classify("How to configure logging?"), "procedural")

    def test_classify_exploratory_query(self):
        """classify should identify exploratory queries."""
        from rag_system.query.classifier import QueryClassifier

        classifier = QueryClassifier()

        # Open-ended exploration
        self.assertEqual(classifier.classify("Tell me about caching"), "exploratory")
        self.assertEqual(classifier.classify("Explain the architecture"), "exploratory")

    def test_classify_troubleshooting_query(self):
        """classify should identify troubleshooting queries."""
        from rag_system.query.classifier import QueryClassifier

        classifier = QueryClassifier()

        # Error/problem questions
        self.assertEqual(classifier.classify("Why is Redis not connecting?"), "troubleshooting")
        self.assertEqual(classifier.classify("Error when starting the service"), "troubleshooting")

    def test_classify_navigational_query(self):
        """classify should identify navigational queries."""
        from rag_system.query.classifier import QueryClassifier

        classifier = QueryClassifier()

        # Looking for specific location
        self.assertEqual(classifier.classify("Where is the config file?"), "navigational")
        self.assertEqual(classifier.classify("Find the API documentation"), "navigational")

    def test_classify_default(self):
        """classify should default to factual for unknown patterns."""
        from rag_system.query.classifier import QueryClassifier

        classifier = QueryClassifier()

        # Ambiguous queries default to factual
        self.assertEqual(classifier.classify("redis configuration"), "factual")


class TestQueryClassifierWithLLM(unittest.TestCase):
    """Test query classification with LLM."""

    def test_classify_with_llm(self):
        """classify_with_llm should use LLM for classification."""
        from rag_system.query.classifier import QueryClassifier

        mock_client = MagicMock()
        mock_client.complete.return_value = "procedural"

        classifier = QueryClassifier(chat_client=mock_client)
        result = classifier.classify_with_llm("How do I set up Redis?")

        self.assertEqual(result, "procedural")
        mock_client.complete.assert_called_once()

    def test_classify_with_llm_handles_error(self):
        """classify_with_llm should fall back to rule-based on error."""
        from rag_system.query.classifier import QueryClassifier

        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("API error")

        classifier = QueryClassifier(chat_client=mock_client)
        result = classifier.classify_with_llm("How do I set up Redis?")

        # Falls back to rule-based
        self.assertEqual(result, "procedural")


class TestQueryPatterns(unittest.TestCase):
    """Test query pattern matching."""

    def test_detect_question_type(self):
        """detect_question_type should identify question patterns."""
        from rag_system.query.classifier import detect_question_type

        self.assertEqual(detect_question_type("What is X?"), "what")
        self.assertEqual(detect_question_type("How do I X?"), "how")
        self.assertEqual(detect_question_type("Why is X?"), "why")
        self.assertEqual(detect_question_type("Where is X?"), "where")
        self.assertEqual(detect_question_type("When should I X?"), "when")
        self.assertEqual(detect_question_type("X Y Z"), None)

    def test_has_error_keywords(self):
        """has_error_keywords should detect error-related words."""
        from rag_system.query.classifier import has_error_keywords

        self.assertTrue(has_error_keywords("Redis error connection"))
        self.assertTrue(has_error_keywords("Why is it failing?"))
        self.assertTrue(has_error_keywords("Not working properly"))
        self.assertFalse(has_error_keywords("How to install Redis"))


if __name__ == '__main__':
    unittest.main()
