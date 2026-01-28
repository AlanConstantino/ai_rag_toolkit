"""Citation validator module for the RAG system.

Extracts and validates chunk citations from generated answers to catch hallucinations.
"""

import re
from typing import List, Set


class CitationValidationResult:
    """Result of citation validation."""

    def __init__(self, is_valid: bool, valid_citations: List[int],
                 invalid_citations: List[int], citation_count: int):
        """Initialize validation result.

        Args:
            is_valid: True if all citations are valid (or no citations present).
            valid_citations: List of chunk IDs that exist in retrieved chunks.
            invalid_citations: List of chunk IDs that don't exist (fabricated).
            citation_count: Total number of citations found.
        """
        self.is_valid = is_valid
        self.valid_citations = valid_citations
        self.invalid_citations = invalid_citations
        self.citation_count = citation_count


def extract_citations(text: str) -> List[int]:
    """Extract chunk citation IDs from text.

    Finds all [CHUNK:id] patterns and returns the IDs.

    Args:
        text: Text containing citations.

    Returns:
        List of chunk IDs in order of appearance.
    """
    # Match [CHUNK:id] with optional spaces: [CHUNK:5], [ CHUNK: 10 ], etc.
    pattern = r'\[\s*CHUNK\s*:\s*(\d+)\s*\]'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [int(m) for m in matches]


def validate_citations(answer: str, valid_chunk_ids: Set[int]) -> CitationValidationResult:
    """Validate that all citations in an answer reference real chunks.

    Args:
        answer: Generated answer text containing citations.
        valid_chunk_ids: Set of chunk IDs that were actually retrieved.

    Returns:
        CitationValidationResult with validation details.
    """
    citations = extract_citations(answer)

    if not citations:
        # No citations means no fabrication (valid by default)
        return CitationValidationResult(
            is_valid=True,
            valid_citations=[],
            invalid_citations=[],
            citation_count=0
        )

    valid = []
    invalid = []

    for citation_id in citations:
        if citation_id in valid_chunk_ids:
            valid.append(citation_id)
        else:
            invalid.append(citation_id)

    return CitationValidationResult(
        is_valid=len(invalid) == 0,
        valid_citations=valid,
        invalid_citations=invalid,
        citation_count=len(citations)
    )
