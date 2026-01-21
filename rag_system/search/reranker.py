"""Reranker module for the RAG system.

Provides final reranking with contextual boosts.
"""

from typing import Dict, List, Tuple, Optional

from rag_system.database import get_connection
from rag_system.utils import tokenize


def rerank(results: List[Tuple[int, float]],
           boost_factors: Optional[Dict[int, float]] = None,
           top_k: int = None) -> List[Tuple[int, float]]:
    """Rerank results with optional boost factors.

    Args:
        results: List of (chunk_id, score).
        boost_factors: Dict mapping chunk_id to boost multiplier.
        top_k: Maximum results to return.

    Returns:
        Reranked list of (chunk_id, score).
    """
    boost_factors = boost_factors or {}

    reranked = []
    for chunk_id, score in results:
        boost = boost_factors.get(chunk_id, 1.0)
        reranked.append((chunk_id, score * boost))

    # Sort by score descending
    reranked.sort(key=lambda x: x[1], reverse=True)

    if top_k:
        reranked = reranked[:top_k]

    return reranked


class Reranker:
    """Contextual reranker using heading paths and query terms."""

    HEADING_MATCH_BOOST = 1.3  # Boost for heading path matches

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the reranker.

        Args:
            db_path: Path to the SQLite database (optional).
        """
        self.db_path = db_path
        self._chunk_headings = None

    def _load_chunk_headings(self) -> Dict[int, str]:
        """Load chunk heading paths from database.

        Returns:
            Dict mapping chunk_id to heading_path.
        """
        if not self.db_path:
            return {}

        if self._chunk_headings is not None:
            return self._chunk_headings

        conn = get_connection(self.db_path)
        try:
            cursor = conn.execute("SELECT id, heading_path FROM chunks")
            self._chunk_headings = {
                row['id']: row['heading_path'] or ''
                for row in cursor
            }
            return self._chunk_headings
        finally:
            conn.close()

    def _calculate_heading_boost(self, heading_path: str,
                                 query_terms: List[str]) -> float:
        """Calculate boost based on heading path match.

        Args:
            heading_path: Chunk's heading path.
            query_terms: Tokenized query terms.

        Returns:
            Boost factor (1.0 = no boost).
        """
        if not heading_path or not query_terms:
            return 1.0

        heading_lower = heading_path.lower()
        matches = sum(1 for term in query_terms if term in heading_lower)

        if matches > 0:
            return self.HEADING_MATCH_BOOST
        return 1.0

    def clear_cache(self) -> None:
        """Clear the heading cache."""
        self._chunk_headings = None

    def rerank(self, results: List[Tuple[int, float]], query: str,
               top_k: int = None) -> List[Tuple[int, float]]:
        """Rerank results with contextual boosts.

        Args:
            results: List of (chunk_id, score).
            query: Original query text.
            top_k: Maximum results to return.

        Returns:
            Reranked list of (chunk_id, score).
        """
        if not self.db_path:
            # No database, just sort by score
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            return sorted_results[:top_k] if top_k else sorted_results

        chunk_headings = self._load_chunk_headings()
        query_terms = tokenize(query)

        # Calculate boosts based on heading matches
        boost_factors = {}
        for chunk_id, _ in results:
            heading_path = chunk_headings.get(chunk_id, '')
            boost_factors[chunk_id] = self._calculate_heading_boost(
                heading_path, query_terms
            )

        return rerank(results, boost_factors, top_k)
