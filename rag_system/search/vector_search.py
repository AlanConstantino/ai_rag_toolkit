"""Vector search module for the RAG system.

Implements cosine similarity search over embeddings.

Memory Requirements:
    - Each embedding (768 dimensions) uses ~6KB in Python
    - 10,000 embeddings = ~60MB memory
    - 100,000 embeddings = ~600MB memory
    - Use max_embeddings parameter to limit memory usage for large datasets

Performance Notes:
    - First search triggers loading embeddings from database
    - Subsequent searches use cached embeddings
    - Use clear_cache() after indexing new documents
    - For very large datasets, consider setting max_embeddings
"""

import hashlib
import json
import math
import os
from typing import List, Tuple, Optional, Any, Dict, Iterator

from rag_system.database import get_connection, get_all_chunks_with_embeddings
from rag_system.utils import get_logger

logger = get_logger(__name__)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity (-1.0 to 1.0), or 0.0 for invalid inputs.
    """
    # Handle empty or mismatched dimensions
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        logger.warning(f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}")
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class VectorIndex:
    """Serializable vector index for efficient similarity search.

    This index can be saved to and loaded from disk, avoiding the need to
    reload embeddings from the database on every search.
    """

    INDEX_VERSION = 1

    def __init__(self):
        """Initialize an empty vector index."""
        self._embeddings: Dict[int, List[float]] = {}
        self._content_hash: Optional[str] = None
        self._chunk_count: int = 0

    def add(self, chunk_id: int, embedding: List[float]) -> None:
        """Add an embedding to the index.

        Args:
            chunk_id: Chunk ID.
            embedding: Embedding vector.
        """
        self._embeddings[chunk_id] = embedding
        self._chunk_count = len(self._embeddings)

    def remove(self, chunk_id: int) -> None:
        """Remove an embedding from the index.

        Args:
            chunk_id: Chunk ID to remove.
        """
        self._embeddings.pop(chunk_id, None)
        self._chunk_count = len(self._embeddings)

    def get(self, chunk_id: int) -> Optional[List[float]]:
        """Get an embedding by chunk ID.

        Args:
            chunk_id: Chunk ID.

        Returns:
            Embedding vector or None.
        """
        return self._embeddings.get(chunk_id)

    def items(self) -> Iterator[Tuple[int, List[float]]]:
        """Iterate over all embeddings.

        Yields:
            (chunk_id, embedding) tuples.
        """
        for chunk_id, embedding in self._embeddings.items():
            yield chunk_id, embedding

    @property
    def count(self) -> int:
        """Number of embeddings in the index."""
        return self._chunk_count

    @property
    def content_hash(self) -> Optional[str]:
        """Hash of the content used to build this index."""
        return self._content_hash

    @content_hash.setter
    def content_hash(self, value: str) -> None:
        """Set the content hash."""
        self._content_hash = value

    def clear(self) -> None:
        """Clear all embeddings from the index."""
        self._embeddings.clear()
        self._chunk_count = 0
        self._content_hash = None

    def save(self, filepath: str) -> None:
        """Save the index to a file.

        Args:
            filepath: Path to save the index.

        Raises:
            IOError: If save fails.
        """
        data = {
            'version': self.INDEX_VERSION,
            'content_hash': self._content_hash,
            'embeddings': self._embeddings
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            logger.info(f"Saved vector index with {self._chunk_count} embeddings to {filepath}")
        except (IOError, OSError) as e:
            logger.error(f"Failed to save vector index: {e}")
            raise

    @classmethod
    def load(cls, filepath: str) -> Optional['VectorIndex']:
        """Load an index from a file.

        Args:
            filepath: Path to the index file.

        Returns:
            Loaded VectorIndex or None if file doesn't exist or is invalid.
        """
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if data.get('version') != cls.INDEX_VERSION:
                logger.warning(f"Index version mismatch, ignoring cached index")
                return None

            index = cls()
            index._content_hash = data.get('content_hash')

            # Convert string keys back to integers
            embeddings_data = data.get('embeddings', {})
            for chunk_id_str, embedding in embeddings_data.items():
                index._embeddings[int(chunk_id_str)] = embedding

            index._chunk_count = len(index._embeddings)
            logger.info(f"Loaded vector index with {index._chunk_count} embeddings from {filepath}")
            return index

        except (json.JSONDecodeError, IOError, OSError, ValueError) as e:
            logger.warning(f"Failed to load vector index: {e}")
            return None


class VectorSearch:
    """Performs vector similarity search over chunk embeddings.

    Memory Requirements:
        The search loads all embeddings into memory for efficient search.
        Each embedding (768 dimensions) uses approximately 6KB.
        For large datasets, use max_embeddings to limit memory usage.

    Example:
        # Basic usage
        search = VectorSearch(db_path)
        results = search.search(query_embedding, top_k=10)

        # With memory limit
        search = VectorSearch(db_path, max_embeddings=10000)

        # With index persistence
        search = VectorSearch(db_path, index_path='index.json')
        search.search(query_embedding)  # Loads/creates index
        # Index is automatically saved when built

    Attributes:
        db_path: Path to the SQLite database.
        vector_client: Optional vector API client.
        max_embeddings: Maximum number of embeddings to load.
        index_path: Path for persisting the vector index.
    """

    def __init__(self, db_path: str, vector_client: Optional[Any] = None,
                 max_embeddings: Optional[int] = None,
                 index_path: Optional[str] = None):
        """Initialize the vector searcher.

        Args:
            db_path: Path to the SQLite database.
            vector_client: Optional vector API client for query embedding.
            max_embeddings: Maximum number of embeddings to load into memory.
                If None, loads all embeddings. For large datasets, set this
                to limit memory usage (e.g., 10000 for ~60MB).
            index_path: Path to save/load the vector index. If provided,
                the index will be persisted to disk for faster subsequent loads.
        """
        self.db_path = db_path
        self.vector_client = vector_client
        self.max_embeddings = max_embeddings
        self.index_path = index_path
        self._index: Optional[VectorIndex] = None

    def _compute_content_hash(self) -> str:
        """Compute a hash of the current database content.

        Returns:
            SHA256 hash of chunk IDs and embedding update timestamps.
        """
        conn = get_connection(self.db_path)
        try:
            # Get chunk count and latest update
            cursor = conn.execute("""
                SELECT COUNT(*) as count,
                       MAX(id) as max_id
                FROM chunks
                WHERE embedding_json IS NOT NULL
            """)
            row = cursor.fetchone()
            count = row['count'] or 0
            max_id = row['max_id'] or 0

            # Create hash from content signature
            signature = f"{count}:{max_id}:{self.db_path}"
            return hashlib.sha256(signature.encode()).hexdigest()
        finally:
            conn.close()

    def _load_index(self) -> VectorIndex:
        """Load or build the vector index.

        Returns:
            VectorIndex with embeddings loaded.
        """
        if self._index is not None:
            return self._index

        current_hash = self._compute_content_hash()

        # Try to load from disk if index_path is specified
        if self.index_path:
            loaded_index = VectorIndex.load(self.index_path)
            if loaded_index and loaded_index.content_hash == current_hash:
                self._index = loaded_index
                return self._index
            elif loaded_index:
                logger.info("Database content changed, rebuilding index")

        # Build index from database
        self._index = self._build_index()
        self._index.content_hash = current_hash

        # Save to disk if index_path is specified
        if self.index_path:
            try:
                self._index.save(self.index_path)
            except IOError:
                logger.warning("Failed to persist vector index, continuing without persistence")

        return self._index

    def _build_index(self) -> VectorIndex:
        """Build the vector index from database.

        Returns:
            VectorIndex with embeddings loaded.
        """
        index = VectorIndex()
        conn = get_connection(self.db_path)

        try:
            chunks = get_all_chunks_with_embeddings(conn)
            loaded_count = 0

            for chunk in chunks:
                if self.max_embeddings and loaded_count >= self.max_embeddings:
                    logger.warning(
                        f"Reached max_embeddings limit ({self.max_embeddings}), "
                        f"some embeddings not loaded"
                    )
                    break

                chunk_id = chunk['id']
                embedding_json = chunk['embedding_json']

                if embedding_json:
                    embedding = json.loads(embedding_json)
                    index.add(chunk_id, embedding)
                    loaded_count += 1

            logger.info(f"Built vector index with {index.count} embeddings")
            return index

        finally:
            conn.close()

    def _load_embeddings(self) -> List[Tuple[int, List[float]]]:
        """Load all chunk embeddings from database.

        This method is kept for backward compatibility.

        Returns:
            List of (chunk_id, embedding) tuples.
        """
        index = self._load_index()
        return list(index.items())

    def clear_cache(self) -> None:
        """Clear the embeddings cache.

        Call this method after indexing new documents to ensure
        the search uses the updated embeddings.
        """
        self._index = None
        logger.debug("Vector search cache cleared")

    def invalidate_if_changed(self) -> bool:
        """Check if database has changed and invalidate cache if so.

        Returns:
            True if cache was invalidated, False if cache is still valid.
        """
        if self._index is None:
            return False

        current_hash = self._compute_content_hash()
        if self._index.content_hash != current_hash:
            self.clear_cache()
            return True
        return False

    def get_memory_usage_estimate(self) -> Dict[str, Any]:
        """Estimate memory usage of the current index.

        Returns:
            Dict with memory usage information.
        """
        index = self._load_index()
        count = index.count

        if count == 0:
            return {
                'embedding_count': 0,
                'estimated_mb': 0,
                'max_embeddings': self.max_embeddings,
                'is_limited': False
            }

        # Get dimension from first embedding
        first_embedding = next(iter(index._embeddings.values()), [])
        dimension = len(first_embedding)

        # Estimate: 8 bytes per float + dict overhead (~100 bytes per entry)
        bytes_per_embedding = dimension * 8 + 100
        total_bytes = count * bytes_per_embedding
        total_mb = total_bytes / (1024 * 1024)

        return {
            'embedding_count': count,
            'dimension': dimension,
            'bytes_per_embedding': bytes_per_embedding,
            'estimated_mb': round(total_mb, 2),
            'max_embeddings': self.max_embeddings,
            'is_limited': self.max_embeddings is not None and count >= self.max_embeddings
        }

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
        index = self._load_index()

        if index.count == 0:
            return []

        # Calculate similarities
        results = []
        for chunk_id, embedding in index.items():
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
