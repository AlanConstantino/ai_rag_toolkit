"""BM25 search module for the RAG system.

Implements BM25 scoring algorithm for lexical search.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple, Set

from rag_system import config
from rag_system.database import (
    get_connection, insert_doc_terms, get_doc_terms,
    update_corpus_stats, get_corpus_stats,
    update_term_doc_frequencies, get_term_doc_frequency
)
from rag_system.utils import get_logger

logger = get_logger(__name__)


# Common English stopwords
STOPWORDS: Set[str] = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why',
    'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there'
}


def tokenize_for_bm25(text: str, remove_stopwords: bool = False) -> List[str]:
    """Tokenize text for BM25 indexing/search.

    Args:
        text: Text to tokenize.
        remove_stopwords: Whether to remove common stopwords.

    Returns:
        List of lowercase tokens.
    """
    # Remove punctuation and split
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    return tokens


def bm25_score(query_terms: List[str], doc_term_freqs: Dict[str, int],
               doc_length: int, avg_doc_length: float,
               doc_frequencies: Dict[str, int], total_docs: int,
               k1: float = None, b: float = None) -> float:
    """Calculate BM25 score for a document given a query.

    Args:
        query_terms: List of query tokens.
        doc_term_freqs: Dict of {term: frequency} for this document.
        doc_length: Number of terms in this document.
        avg_doc_length: Average document length across corpus.
        doc_frequencies: Dict of {term: num_docs_containing_term}.
        total_docs: Total documents in corpus.
        k1: Term frequency saturation parameter (default from config).
        b: Length normalization parameter (default from config).

    Returns:
        BM25 score.
    """
    k1 = k1 if k1 is not None else config.BM25_K1
    b = b if b is not None else config.BM25_B

    score = 0.0

    for term in query_terms:
        if term not in doc_term_freqs:
            continue

        tf = doc_term_freqs[term]
        df = doc_frequencies.get(term, 0)

        # IDF with smoothing
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

        # TF with saturation and length normalization
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
        tf_component = numerator / denominator

        score += idf * tf_component

    return score


class BM25Index:
    """Builds and maintains BM25 index."""

    def __init__(self, db_path: str):
        """Initialize the BM25 index.

        Args:
            db_path: Path to the SQLite database.
        """
        self.db_path = db_path

    def build(self) -> None:
        """Build BM25 index from all small chunks in database."""
        conn = get_connection(self.db_path)

        try:
            # Get all small chunks
            cursor = conn.execute(
                "SELECT id, content FROM chunks WHERE chunk_type = 'small'"
            )
            chunks = cursor.fetchall()

            if not chunks:
                logger.warning("No chunks to index")
                return

            # Track term frequencies across documents
            term_doc_counts: Dict[str, int] = Counter()
            total_length = 0

            # Process each chunk
            for chunk_id, content in chunks:
                tokens = tokenize_for_bm25(content, remove_stopwords=True)
                term_freqs = Counter(tokens)

                # Store term frequencies for this chunk
                # First clear any existing terms
                conn.execute("DELETE FROM doc_terms WHERE chunk_id = ?", (chunk_id,))
                insert_doc_terms(conn, chunk_id, dict(term_freqs))

                # Update document frequencies
                for term in set(tokens):
                    term_doc_counts[term] += 1

                total_length += len(tokens)

            # Store corpus statistics
            avg_length = total_length / len(chunks) if chunks else 0
            update_corpus_stats(conn, total_docs=len(chunks), avg_doc_length=avg_length)

            # Store term document frequencies
            update_term_doc_frequencies(conn, dict(term_doc_counts))

            logger.info(f"Built BM25 index: {len(chunks)} docs, {len(term_doc_counts)} unique terms")

        finally:
            conn.close()


class BM25Search:
    """Performs BM25 search over indexed chunks."""

    def __init__(self, db_path: str):
        """Initialize the BM25 searcher.

        Args:
            db_path: Path to the SQLite database.
        """
        self.db_path = db_path

    def search(self, query: str, top_k: int = 10,
               chunk_type: str = 'small') -> List[Tuple[int, float]]:
        """Search for chunks matching the query.

        Args:
            query: Search query.
            top_k: Maximum number of results.
            chunk_type: Type of chunks to search ('small' or 'large').

        Returns:
            List of (chunk_id, score) tuples sorted by score descending.
        """
        query_terms = tokenize_for_bm25(query, remove_stopwords=True)

        if not query_terms:
            return []

        conn = get_connection(self.db_path)

        try:
            # Get corpus statistics
            stats = get_corpus_stats(conn)
            if not stats:
                return []

            total_docs = stats['total_docs']
            avg_doc_length = stats['avg_doc_length']

            # Get document frequencies for query terms
            doc_frequencies = {}
            for term in query_terms:
                doc_frequencies[term] = get_term_doc_frequency(conn, term)

            # Get all chunks with matching terms
            # First find chunks that have at least one query term
            placeholders = ','.join(['?' for _ in query_terms])
            cursor = conn.execute(
                f"""SELECT DISTINCT chunk_id FROM doc_terms
                    WHERE term IN ({placeholders})""",
                query_terms
            )
            candidate_chunk_ids = [row[0] for row in cursor.fetchall()]

            if not candidate_chunk_ids:
                return []

            # Score each candidate
            results = []

            for chunk_id in candidate_chunk_ids:
                # Get term frequencies for this chunk
                term_freqs = get_doc_terms(conn, chunk_id)
                doc_length = sum(term_freqs.values())

                score = bm25_score(
                    query_terms, term_freqs, doc_length,
                    avg_doc_length, doc_frequencies, total_docs
                )

                if score > 0:
                    results.append((chunk_id, score))

            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)

            return results[:top_k]

        finally:
            conn.close()
