"""Tests for answer generator grounding safeguards.

Tests the citation format requirements and fallback responses.
"""

import unittest
from unittest.mock import Mock
from rag_system.query.answer_generator import (
    AnswerGenerator,
    build_grounded_answer_prompt,
    format_fallback_response,
    GROUNDED_PROMPT_TEMPLATE
)


class TestGroundedPromptTemplate(unittest.TestCase):
    """Tests for the grounded prompt template."""

    def test_template_requires_citation_format(self):
        """Template must include citation format requirement."""
        self.assertIn('[CHUNK:id]', GROUNDED_PROMPT_TEMPLATE)
        self.assertIn('citation', GROUNDED_PROMPT_TEMPLATE.lower())

    def test_template_includes_grounding_instruction(self):
        """Template must instruct to use only provided context."""
        template_lower = GROUNDED_PROMPT_TEMPLATE.lower()
        self.assertTrue(
            'only use' in template_lower or
            'only from' in template_lower or
            'must come from' in template_lower
        )


class TestBuildGroundedAnswerPrompt(unittest.TestCase):
    """Tests for build_grounded_answer_prompt function."""

    def test_includes_chunk_ids_in_context(self):
        """Context should include chunk IDs for citation."""
        chunks = [
            {'id': 5, 'content': 'First chunk content'},
            {'id': 10, 'content': 'Second chunk content'}
        ]
        prompt = build_grounded_answer_prompt("test query", chunks)

        self.assertIn('[CHUNK:5]', prompt)
        self.assertIn('[CHUNK:10]', prompt)
        self.assertIn('First chunk content', prompt)
        self.assertIn('Second chunk content', prompt)

    def test_includes_query(self):
        """Prompt should include the query."""
        chunks = [{'id': 1, 'content': 'content'}]
        prompt = build_grounded_answer_prompt("How do I configure timeouts?", chunks)

        self.assertIn("How do I configure timeouts?", prompt)

    def test_empty_chunks_handled(self):
        """Handle empty chunks list gracefully."""
        prompt = build_grounded_answer_prompt("test query", [])
        self.assertIn("test query", prompt)


class TestFormatFallbackResponse(unittest.TestCase):
    """Tests for format_fallback_response function."""

    def test_fallback_includes_relevant_chunks(self):
        """Fallback shows relevant passages instead of 'I don't know'."""
        chunks = [
            {
                'id': 12,
                'content': 'The timeout configuration can be set in config.yaml...',
                'page_url': 'docs/configuration.md',
                'page_title': 'Configuration Guide'
            },
            {
                'id': 45,
                'content': 'Default values are applied when no explicit setting...',
                'page_url': 'docs/defaults.md',
                'page_title': 'Default Values'
            }
        ]
        response = format_fallback_response(chunks)

        # Should not say "I don't know"
        self.assertNotIn("I don't know", response)

        # Should include the chunks
        self.assertIn('[CHUNK:12]', response)
        self.assertIn('[CHUNK:45]', response)
        self.assertIn('timeout configuration', response)
        self.assertIn('Default values', response)

        # Should include sources
        self.assertIn('docs/configuration.md', response)

    def test_fallback_with_empty_chunks(self):
        """Fallback with no chunks gives helpful message."""
        response = format_fallback_response([])

        # Should still be helpful, not just "I don't know"
        self.assertIn("couldn't find", response.lower())

    def test_fallback_message_is_constructive(self):
        """Fallback should keep users moving forward."""
        chunks = [
            {'id': 1, 'content': 'Some content here', 'page_url': 'docs/test.md', 'page_title': 'Test'}
        ]
        response = format_fallback_response(chunks)

        # Should be constructive
        self.assertTrue(
            'relevant' in response.lower() or
            'passage' in response.lower() or
            'section' in response.lower()
        )


class TestAnswerGeneratorWithGrounding(unittest.TestCase):
    """Tests for AnswerGenerator with grounding safeguards."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_chat_client = Mock()
        self.generator = AnswerGenerator(self.mock_chat_client)

    def test_generate_grounded_uses_chunk_ids(self):
        """generate_grounded should pass chunk IDs to the LLM."""
        chunks = [
            {'id': 5, 'content': 'Configuration content', 'page_url': 'docs/config.md', 'page_title': 'Config'},
            {'id': 10, 'content': 'Default settings', 'page_url': 'docs/defaults.md', 'page_title': 'Defaults'}
        ]

        self.mock_chat_client.complete.return_value = "The config [CHUNK:5] shows..."

        result = self.generator.generate_grounded("How to configure?", chunks)

        # Verify the prompt sent to LLM includes chunk IDs
        call_args = self.mock_chat_client.complete.call_args[0][0]
        self.assertIn('[CHUNK:5]', call_args)
        self.assertIn('[CHUNK:10]', call_args)

    def test_generate_grounded_validates_citations(self):
        """generate_grounded should validate citations in response."""
        chunks = [
            {'id': 5, 'content': 'Real content', 'page_url': 'docs/real.md', 'page_title': 'Real'}
        ]

        # LLM returns a fabricated citation
        self.mock_chat_client.complete.return_value = "According to [CHUNK:999], the answer is..."

        result = self.generator.generate_grounded("test query", chunks)

        # Result should flag the invalid citation
        self.assertFalse(result['citations_valid'])
        self.assertIn(999, result['invalid_citations'])

    def test_generate_grounded_returns_fallback_on_low_confidence(self):
        """Use fallback response when confidence is too low."""
        chunks = [
            {'id': 1, 'content': 'Some marginally relevant content', 'page_url': 'docs/test.md', 'page_title': 'Test'}
        ]

        self.generator.generate_grounded("test", chunks, confidence=0.2)

        # With very low confidence, should return fallback-style response

    def test_generate_grounded_result_structure(self):
        """Verify the result structure from generate_grounded."""
        chunks = [
            {'id': 5, 'content': 'Content here', 'page_url': 'docs/test.md', 'page_title': 'Test'}
        ]
        self.mock_chat_client.complete.return_value = "Answer with [CHUNK:5] citation."

        result = self.generator.generate_grounded("query", chunks)

        # Check result structure
        self.assertIn('answer', result)
        self.assertIn('citations_valid', result)
        self.assertIn('valid_citations', result)
        self.assertIn('invalid_citations', result)
        self.assertIn('used_fallback', result)


class TestIntegrationGroundingWorkflow(unittest.TestCase):
    """Integration tests for the full grounding workflow."""

    def test_end_to_end_valid_answer(self):
        """Full workflow with valid citations."""
        mock_chat = Mock()
        mock_chat.complete.return_value = (
            "Based on the documentation [CHUNK:5], the timeout is 30 seconds. "
            "You can also configure it in config.yaml [CHUNK:10]."
        )

        generator = AnswerGenerator(mock_chat)
        chunks = [
            {'id': 5, 'content': 'Timeout is 30 seconds by default', 'page_url': 'docs/timeout.md', 'page_title': 'Timeout'},
            {'id': 10, 'content': 'config.yaml contains settings', 'page_url': 'docs/config.md', 'page_title': 'Config'}
        ]

        result = generator.generate_grounded("What is the timeout?", chunks)

        self.assertTrue(result['citations_valid'])
        self.assertEqual(sorted(result['valid_citations']), [5, 10])
        self.assertEqual(result['invalid_citations'], [])
        self.assertFalse(result['used_fallback'])

    def test_end_to_end_fabricated_citation(self):
        """Full workflow catches fabricated citations."""
        mock_chat = Mock()
        mock_chat.complete.return_value = (
            "According to [CHUNK:5] and [CHUNK:999], the answer is clear."
        )

        generator = AnswerGenerator(mock_chat)
        chunks = [
            {'id': 5, 'content': 'Real content', 'page_url': 'docs/real.md', 'page_title': 'Real'}
        ]

        result = generator.generate_grounded("test query", chunks)

        self.assertFalse(result['citations_valid'])
        self.assertEqual(result['valid_citations'], [5])
        self.assertEqual(result['invalid_citations'], [999])


if __name__ == '__main__':
    unittest.main()
