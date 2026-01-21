"""Vector search module for the RAG system.

Implements cosine similarity search over embeddings.
"""

import json
import math
from typing import List, Tuple, Optional, Any

from rag_system.database import get_connection, get_all_chunks_with_embeddings
from rag_system.utils import get_logger

logger = get_logger(__name__)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity (-1.0 to 1.0).
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class VectorSearch:
    """Performs vector similarity search over chunk embeddings."""

    def __init__(self, db_path: str, vector_client: Optional[Any] = None):
        """Initialize the vector searcher.

        Args:
            db_path: Path to the SQLite database.
            vector_client: Optional vector API client for query embedding.
        """
        self.db_path = db_path
        self.vector_client = vector_client
        self._embeddings_cache = None

    def _load_embeddings(self) -> List[Tuple[int, List[float]]]:
        """Load all chunk embeddings from database.

        Returns:
            List of (chunk_id, embedding) tuples.
        """
        if self._embeddings_cache is not None:
            return self._embeddings_cache

        conn = get_connection(self.db_path)

        try:
            chunks = get_all_chunks_with_embeddings(conn)
            embeddings = []

            for chunk in chunks:
                chunk_id = chunk['id']
                embedding_json = chunk['embedding_json']

                if embedding_json:
                    embedding = json.loads(embedding_json)
                    embeddings.append((chunk_id, embedding))

            self._embeddings_cache = embeddings
            return embeddings

        finally:
            conn.close()

    def clear_cache(self) -> None:
        """Clear the embeddings cache."""
        self._embeddings_cache = None

    def search(self, query_embedding: List[float], top_k: int = 10,
               threshold: float = 0.0) -> List[Tuple[int, float]]:
        """Search for chunks similar to the query embedding.

        Args:
            query_embedding: Query vector.
            top_k: Maximum number of results.
            threshold: Minimum similarity threshold.

        Returns:
            List of (chunk_id, similarity) tuples sorted by similarity descending.
        """
        embeddings = self._load_embeddings()

        if not embeddings:
            return []

        # Calculate similarities
        results = []
        for chunk_id, embedding in embeddings:
            similarity = cosine_similarity(query_embedding, embedding)
            if similarity >= threshold:
                results.append((chunk_id, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def search_text(self, query: str, top_k: int = 10,
                    threshold: float = 0.0) -> List[Tuple[int, float]]:
        """Search using a text query (requires vector client).

        Args:
            query: Text query.
            top_k: Maximum number of results.
            threshold: Minimum similarity threshold.

        Returns:
            List of (chunk_id, similarity) tuples sorted by similarity descending.

        Raises:
            ValueError: If no vector client is configured.
        """
        if not self.vector_client:
            raise ValueError("Vector client required for text search")

        query_embedding = self.vector_client.get_embedding(query)
        return self.search(query_embedding, top_k, threshold)
