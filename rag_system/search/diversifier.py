"""Diversifier module for the RAG system.

Ensures search results are diverse across pages.
"""

from typing import Dict, List, Tuple, Any

from rag_system import config
from rag_system.database import get_connection


def diversify_results(ranked_results: List[Tuple[int, float]],
                      chunk_to_page: Dict[int, Any],
                      max_per_page: int = None,
                      top_k: int = None) -> List[Tuple[int, float]]:
    """Ensure results are diverse across pages.

    Args:
        ranked_results: List of (chunk_id, score) sorted by score.
        chunk_to_page: Dict mapping chunk_id to page_id.
        max_per_page: Maximum chunks from any single page.
        top_k: Number of results to return.

    Returns:
        Diversified list of (chunk_id, score).
    """
    max_per_page = max_per_page if max_per_page is not None else config.MAX_CHUNKS_PER_PAGE
    top_k = top_k if top_k is not None else config.TOP_K_FINAL

    diversified = []
    page_counts: Dict[Any, int] = {}

    for chunk_id, score in ranked_results:
        page_id = chunk_to_page.get(chunk_id)
        current_count = page_counts.get(page_id, 0)

        if current_count < max_per_page:
            diversified.append((chunk_id, score))
            page_counts[page_id] = current_count + 1

        if len(diversified) >= top_k:
            break

    return diversified


class Diversifier:
    """Diversifies results using database chunk-page mapping."""

    def __init__(self, db_path: str):
        """Initialize the diversifier.

        Args:
            db_path: Path to the SQLite database.
        """
        self.db_path = db_path
        self._chunk_to_page = None

    def _load_chunk_to_page(self) -> Dict[int, int]:
        """Load chunk to page mapping from database.

        Returns:
            Dict mapping chunk_id to page_id.
        """
        if self._chunk_to_page is not None:
            return self._chunk_to_page

        conn = get_connection(self.db_path)
        try:
            cursor = conn.execute("SELECT id, page_id FROM chunks")
            self._chunk_to_page = {row['id']: row['page_id'] for row in cursor}
            return self._chunk_to_page
        finally:
            conn.close()

    def clear_cache(self) -> None:
        """Clear the chunk-to-page cache."""
        self._chunk_to_page = None

    def diversify(self, results: List[Tuple[int, float]],
                  max_per_page: int = None,
                  top_k: int = None) -> List[Tuple[int, float]]:
        """Diversify search results.

        Args:
            results: List of (chunk_id, score) sorted by score.
            max_per_page: Maximum chunks per page.
            top_k: Number of results to return.

        Returns:
            Diversified list of (chunk_id, score).
        """
        chunk_to_page = self._load_chunk_to_page()
        return diversify_results(results, chunk_to_page, max_per_page, top_k)
