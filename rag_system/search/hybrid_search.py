"""Hybrid search module for the RAG system.

Combines vector and BM25 search with score normalization.
"""

from typing import List, Tuple, Optional, Dict

from rag_system import config
from rag_system.utils import get_logger

logger = get_logger(__name__)


def normalize_scores(results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Normalize scores to 0-1 range.

    Args:
        results: List of (chunk_id, score) tuples.

    Returns:
        List of (chunk_id, normalized_score) tuples.
    """
    if not results:
        return []

    scores = [score for _, score in results]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [(chunk_id, 1.0) for chunk_id, _ in results]

    return [
        (chunk_id, (score - min_score) / (max_score - min_score))
        for chunk_id, score in results
    ]


def merge_results(vector_results: List[Tuple[int, float]],
                  bm25_results: List[Tuple[int, float]],
                  vector_weight: float = None,
                  bm25_weight: float = None) -> List[Tuple[int, float]]:
    """Merge and weight vector and BM25 results.

    Args:
        vector_results: Results from vector search (chunk_id, score).
        bm25_results: Results from BM25 search (chunk_id, score).
        vector_weight: Weight for vector scores (default from config).
        bm25_weight: Weight for BM25 scores (default from config).

    Returns:
        Merged list of (chunk_id, combined_score) sorted descending.
    """
    vector_weight = vector_weight if vector_weight is not None else config.VECTOR_WEIGHT
    bm25_weight = bm25_weight if bm25_weight is not None else config.BM25_WEIGHT

    # Normalize scores
    vector_normalized = dict(normalize_scores(vector_results))
    bm25_normalized = dict(normalize_scores(bm25_results))

    # Combine all chunk IDs
    all_chunk_ids = set(vector_normalized.keys()) | set(bm25_normalized.keys())

    # Calculate combined scores
    combined = {}
    for chunk_id in all_chunk_ids:
        v_score = vector_normalized.get(chunk_id, 0.0)
        b_score = bm25_normalized.get(chunk_id, 0.0)
        combined[chunk_id] = (v_score * vector_weight) + (b_score * bm25_weight)

    # Sort by combined score descending
    results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return results


class HybridSearch:
    """Performs hybrid search combining vector and BM25."""

    def __init__(self, vector_search, bm25_search,
                 vector_weight: float = None, bm25_weight: float = None):
        """Initialize hybrid search.

        Args:
            vector_search: VectorSearch instance.
            bm25_search: BM25Search instance.
            vector_weight: Weight for vector scores.
            bm25_weight: Weight for BM25 scores.
        """
        self.vector_search = vector_search
        self.bm25_search = bm25_search
        self.vector_weight = vector_weight if vector_weight is not None else config.VECTOR_WEIGHT
        self.bm25_weight = bm25_weight if bm25_weight is not None else config.BM25_WEIGHT

    def search(self, query_embedding: List[float], query_text: str,
               top_k: int = 20) -> List[Tuple[int, float]]:
        """Perform hybrid search.

        Args:
            query_embedding: Query embedding vector.
            query_text: Query text for BM25.
            top_k: Number of results from each search method.

        Returns:
            Merged list of (chunk_id, score) sorted descending.
        """
        # Get results from both methods
        vector_results = self.vector_search.search(query_embedding, top_k=top_k)
        bm25_results = self.bm25_search.search(query_text, top_k=top_k)

        logger.debug(f"Vector results: {len(vector_results)}, BM25 results: {len(bm25_results)}")

        # Merge results
        return merge_results(
            vector_results, bm25_results,
            self.vector_weight, self.bm25_weight
        )
