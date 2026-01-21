"""Tests for the answer generator module."""

import unittest
from unittest.mock import MagicMock


class TestAnswerGenerator(unittest.TestCase):
    """Test answer generation."""

    def test_generate_answer(self):
        """generate should return LLM answer."""
        from rag_system.query.answer_generator import AnswerGenerator

        mock_client = MagicMock()
        mock_client.complete.return_value = "Redis is an in-memory cache."

        generator = AnswerGenerator(chat_client=mock_client)
        answer = generator.generate("What is Redis?", "Redis is a cache...")

        self.assertEqual(answer, "Redis is an in-memory cache.")
        mock_client.complete.assert_called_once()

    def test_generate_handles_error(self):
        """generate should handle API errors."""
        from rag_system.query.answer_generator import AnswerGenerator

        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("API error")

        generator = AnswerGenerator(chat_client=mock_client)
        answer = generator.generate("What is Redis?", "context")

        self.assertIn("unable to generate", answer.lower())

    def test_generate_without_client(self):
        """generate should fail gracefully without client."""
        from rag_system.query.answer_generator import AnswerGenerator

        generator = AnswerGenerator()
        answer = generator.generate("What is Redis?", "context")

        self.assertIn("not configured", answer.lower())


class TestAnswerWithConfidence(unittest.TestCase):
    """Test answer generation with confidence."""

    def test_generate_with_low_confidence(self):
        """generate_with_confidence should indicate uncertainty."""
        from rag_system.query.answer_generator import AnswerGenerator

        mock_client = MagicMock()
        mock_client.complete.return_value = "Redis might be a cache."

        generator = AnswerGenerator(chat_client=mock_client)
        result = generator.generate_with_confidence(
            "What is Redis?",
            "context",
            confidence=0.3
        )

        self.assertIn('answer', result)
        self.assertIn('disclaimer', result)
        self.assertTrue(len(result['disclaimer']) > 0)

    def test_generate_with_high_confidence(self):
        """generate_with_confidence should not add disclaimer for high confidence."""
        from rag_system.query.answer_generator import AnswerGenerator

        mock_client = MagicMock()
        mock_client.complete.return_value = "Redis is a cache."

        generator = AnswerGenerator(chat_client=mock_client)
        result = generator.generate_with_confidence(
            "What is Redis?",
            "context",
            confidence=0.9
        )

        self.assertIn('answer', result)
        # Disclaimer should be empty or None for high confidence
        self.assertFalse(result.get('disclaimer'))


class TestPromptBuilding(unittest.TestCase):
    """Test prompt building for answers."""

    def test_build_answer_prompt(self):
        """build_answer_prompt should create complete prompt."""
        from rag_system.query.answer_generator import build_answer_prompt

        prompt = build_answer_prompt("What is Redis?", "Redis is a cache.")

        self.assertIn("What is Redis?", prompt)
        self.assertIn("Redis is a cache", prompt)

    def test_build_answer_prompt_with_type(self):
        """build_answer_prompt should adapt to query type."""
        from rag_system.query.answer_generator import build_answer_prompt

        procedural = build_answer_prompt(
            "How to install Redis?",
            "context",
            query_type='procedural'
        )
        self.assertIn('step', procedural.lower())

        factual = build_answer_prompt(
            "What is Redis?",
            "context",
            query_type='factual'
        )
        self.assertIn('fact', factual.lower())


class TestAnswerFormatting(unittest.TestCase):
    """Test answer formatting."""

    def test_format_answer_basic(self):
        """format_answer should return clean answer."""
        from rag_system.query.answer_generator import format_answer

        answer = format_answer("  Redis is a cache.  ")
        self.assertEqual(answer, "Redis is a cache.")

    def test_format_answer_with_sources(self):
        """format_answer should append sources when provided."""
        from rag_system.query.answer_generator import format_answer

        sources = [
            {'title': 'Redis Docs', 'url': 'http://redis.io'}
        ]
        answer = format_answer("Redis is a cache.", sources=sources)

        self.assertIn("Redis is a cache", answer)
        self.assertIn("Redis Docs", answer)

    def test_format_answer_removes_prefix(self):
        """format_answer should remove common prefixes."""
        from rag_system.query.answer_generator import format_answer

        answer = format_answer("Answer: Redis is a cache.")
        self.assertEqual(answer, "Redis is a cache.")


class TestNoAnswerHandling(unittest.TestCase):
    """Test handling when no answer is possible."""

    def test_generate_no_context(self):
        """generate should handle empty context."""
        from rag_system.query.answer_generator import AnswerGenerator

        generator = AnswerGenerator()
        answer = generator.generate("What is Redis?", "")

        self.assertIn("enough information", answer.lower())

    def test_get_fallback_response(self):
        """get_fallback_response should provide helpful message."""
        from rag_system.query.answer_generator import get_fallback_response

        response = get_fallback_response("What is Redis?")

        self.assertIn("Redis", response)
        self.assertTrue(len(response) > 20)


if __name__ == '__main__':
    unittest.main()
