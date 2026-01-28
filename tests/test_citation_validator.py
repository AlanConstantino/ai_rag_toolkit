"""Tests for citation validator module.

Validates that citations in generated answers reference real chunks.
"""

import unittest
from rag_system.query.citation_validator import (
    extract_citations,
    validate_citations,
    CitationValidationResult
)


class TestExtractCitations(unittest.TestCase):
    """Tests for extract_citations function."""

    def test_extract_single_citation(self):
        """Extract a single citation from text."""
        text = "The timeout is 30 seconds [CHUNK:12]."
        citations = extract_citations(text)
        self.assertEqual(citations, [12])

    def test_extract_multiple_citations(self):
        """Extract multiple citations from text."""
        text = "See [CHUNK:5] for config and [CHUNK:23] for defaults."
        citations = extract_citations(text)
        self.assertEqual(sorted(citations), [5, 23])

    def test_extract_no_citations(self):
        """Return empty list when no citations present."""
        text = "This answer has no citations."
        citations = extract_citations(text)
        self.assertEqual(citations, [])

    def test_extract_duplicate_citations(self):
        """Duplicate citations should be included."""
        text = "[CHUNK:5] and also [CHUNK:5] again."
        citations = extract_citations(text)
        self.assertEqual(citations, [5, 5])

    def test_extract_citation_with_spaces(self):
        """Handle citations with various spacing."""
        text = "[CHUNK: 10] and [ CHUNK:20 ] and [CHUNK:30]."
        citations = extract_citations(text)
        self.assertEqual(sorted(citations), [10, 20, 30])

    def test_extract_citations_multiline(self):
        """Extract citations across multiple lines."""
        text = """First point [CHUNK:1].
        Second point [CHUNK:2].
        Third point [CHUNK:3]."""
        citations = extract_citations(text)
        self.assertEqual(citations, [1, 2, 3])


class TestValidateCitations(unittest.TestCase):
    """Tests for validate_citations function."""

    def test_all_valid_citations(self):
        """All citations reference valid chunk IDs."""
        answer = "The config [CHUNK:5] shows timeout [CHUNK:10]."
        valid_chunk_ids = {5, 10, 15, 20}
        result = validate_citations(answer, valid_chunk_ids)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.valid_citations, [5, 10])
        self.assertEqual(result.invalid_citations, [])
        self.assertEqual(result.citation_count, 2)

    def test_some_invalid_citations(self):
        """Some citations reference non-existent chunks."""
        answer = "See [CHUNK:5] and [CHUNK:999]."
        valid_chunk_ids = {5, 10, 15}
        result = validate_citations(answer, valid_chunk_ids)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.valid_citations, [5])
        self.assertEqual(result.invalid_citations, [999])

    def test_all_invalid_citations(self):
        """All citations are invalid (fabricated)."""
        answer = "According to [CHUNK:888] and [CHUNK:999]."
        valid_chunk_ids = {1, 2, 3}
        result = validate_citations(answer, valid_chunk_ids)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.valid_citations, [])
        self.assertEqual(result.invalid_citations, [888, 999])

    def test_no_citations_is_valid(self):
        """No citations is considered valid (no fabrication)."""
        answer = "The answer with no citations."
        valid_chunk_ids = {1, 2, 3}
        result = validate_citations(answer, valid_chunk_ids)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.citation_count, 0)

    def test_empty_valid_chunks(self):
        """Any citations are invalid when no chunks were retrieved."""
        answer = "See [CHUNK:5]."
        valid_chunk_ids = set()
        result = validate_citations(answer, valid_chunk_ids)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.invalid_citations, [5])


class TestCitationValidationResult(unittest.TestCase):
    """Tests for CitationValidationResult dataclass."""

    def test_result_attributes(self):
        """Verify result has expected attributes."""
        result = CitationValidationResult(
            is_valid=True,
            valid_citations=[1, 2],
            invalid_citations=[],
            citation_count=2
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.valid_citations, [1, 2])
        self.assertEqual(result.invalid_citations, [])
        self.assertEqual(result.citation_count, 2)

    def test_result_with_invalid(self):
        """Result correctly indicates invalid state."""
        result = CitationValidationResult(
            is_valid=False,
            valid_citations=[1],
            invalid_citations=[999],
            citation_count=2
        )
        self.assertFalse(result.is_valid)
        self.assertIn(999, result.invalid_citations)


if __name__ == '__main__':
    unittest.main()
