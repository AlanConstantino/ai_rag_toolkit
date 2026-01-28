"""BM25 search module for the RAG system.

Implements BM25 scoring algorithm for lexical search.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple, Set, Optional

from rag_system import config
from rag_system.database import (
    get_connection, insert_doc_terms, get_doc_terms, get_doc_terms_batch,
    update_corpus_stats, get_corpus_stats,
    update_term_doc_frequencies, get_term_doc_frequency, get_term_doc_frequencies_batch
)
from rag_system.utils import get_logger

logger = get_logger(__name__)


class PorterStemmer:
    """Porter Stemmer implementation for English word stemming.

    Based on the classic algorithm from https://tartarus.org/martin/PorterStemmer/
    Implemented using only Python stdlib.
    """

    def __init__(self):
        self._cache: Dict[str, str] = {}

    def _cons(self, word: str, i: int) -> bool:
        """Check if word[i] is a consonant."""
        if word[i] in 'aeiou':
            return False
        if word[i] == 'y':
            return i == 0 or not self._cons(word, i - 1)
        return True

    def _m(self, word: str) -> int:
        """Count VC sequences (measure) in word."""
        n = 0
        i = 0
        length = len(word)
        while i < length:
            if not self._cons(word, i):
                break
            i += 1
        if i >= length:
            return 0
        i += 1
        while i < length:
            while i < length:
                if self._cons(word, i):
                    break
                i += 1
            if i >= length:
                break
            n += 1
            i += 1
            while i < length:
                if not self._cons(word, i):
                    break
                i += 1
        return n

    def _vowelinstem(self, word: str) -> bool:
        """Check if word contains a vowel."""
        return any(not self._cons(word, i) for i in range(len(word)))

    def _doublec(self, word: str) -> bool:
        """Check if word ends with double consonant."""
        return len(word) >= 2 and word[-1] == word[-2] and self._cons(word, len(word) - 1)

    def _cvc(self, word: str) -> bool:
        """Check if word ends with CVC pattern (consonant-vowel-consonant)."""
        if len(word) < 3:
            return False
        i = len(word) - 1
        if not self._cons(word, i) or self._cons(word, i - 1) or not self._cons(word, i - 2):
            return False
        return word[i] not in 'wxy'

    def _step1ab(self, word: str) -> str:
        """Handle plurals and past participles."""
        if word.endswith('sses'):
            word = word[:-2]
        elif word.endswith('ies'):
            word = word[:-2]
        elif not word.endswith('ss') and word.endswith('s'):
            word = word[:-1]

        if word.endswith('eed'):
            if self._m(word[:-3]) > 0:
                word = word[:-1]
        elif word.endswith('ed'):
            stem = word[:-2]
            if self._vowelinstem(stem):
                word = stem
                word = self._step1b_helper(word)
        elif word.endswith('ing'):
            stem = word[:-3]
            if self._vowelinstem(stem):
                word = stem
                word = self._step1b_helper(word)
        return word

    def _step1b_helper(self, word: str) -> str:
        """Helper for step1ab after removing ed/ing."""
        if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
            word = word + 'e'
        elif self._doublec(word) and word[-1] not in 'lsz':
            word = word[:-1]
        elif self._m(word) == 1 and self._cvc(word):
            word = word + 'e'
        return word

    def _step1c(self, word: str) -> str:
        """Replace y with i when preceded by consonant."""
        if word.endswith('y') and self._vowelinstem(word[:-1]):
            word = word[:-1] + 'i'
        return word

    def _step2(self, word: str) -> str:
        """Map double suffixes to single ones."""
        suffixes = [
            ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'),
            ('anci', 'ance'), ('izer', 'ize'), ('abli', 'able'),
            ('alli', 'al'), ('entli', 'ent'), ('eli', 'e'), ('ousli', 'ous'),
            ('ization', 'ize'), ('ation', 'ate'), ('ator', 'ate'),
            ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
            ('ousness', 'ous'), ('aliti', 'al'), ('iviti', 'ive'), ('biliti', 'ble')
        ]
        for suffix, replacement in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._m(stem) > 0:
                    return stem + replacement
                break
        return word

    def _step3(self, word: str) -> str:
        """Handle derivational suffixes."""
        suffixes = [
            ('icate', 'ic'), ('ative', ''), ('alize', 'al'),
            ('iciti', 'ic'), ('ical', 'ic'), ('ful', ''), ('ness', '')
        ]
        for suffix, replacement in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._m(stem) > 0:
                    return stem + replacement
                break
        return word

    def _step4(self, word: str) -> str:
        """Remove derivational suffixes."""
        suffixes = [
            'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant',
            'ement', 'ment', 'ent', 'ion', 'ou', 'ism', 'ate', 'iti',
            'ous', 'ive', 'ize'
        ]
        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if suffix == 'ion':
                    if stem and stem[-1] in 'st' and self._m(stem) > 1:
                        return stem
                elif self._m(stem) > 1:
                    return stem
                break
        return word

    def _step5(self, word: str) -> str:
        """Remove final e or reduce double l."""
        if word.endswith('e'):
            stem = word[:-1]
            m = self._m(stem)
            if m > 1 or (m == 1 and not self._cvc(stem)):
                word = stem
        if word.endswith('ll') and self._m(word) > 1:
            word = word[:-1]
        return word

    def stem(self, word: str) -> str:
        """Stem a word using Porter algorithm.

        Args:
            word: Word to stem.

        Returns:
            Stemmed word.
        """
        if len(word) <= 2:
            return word

        if word in self._cache:
            return self._cache[word]

        original = word
        word = word.lower()

        word = self._step1ab(word)
        word = self._step1c(word)
        word = self._step2(word)
        word = self._step3(word)
        word = self._step4(word)
        word = self._step5(word)

        self._cache[original] = word
        return word


# Global stemmer instance
_stemmer = PorterStemmer()


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


def tokenize_for_bm25(text: str, remove_stopwords: bool = False,
                      stem: bool = False) -> List[str]:
    """Tokenize text for BM25 indexing/search.

    Args:
        text: Text to tokenize.
        remove_stopwords: Whether to remove common stopwords.
        stem: Whether to apply Porter stemming to tokens.

    Returns:
        List of lowercase tokens.
    """
    # Split camelCase: "camelCase" -> "camel Case"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Split snake_case and kebab-case: "snake_case" -> "snake case"
    text = re.sub(r'[_-]', ' ', text)

    # Remove punctuation and split
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]

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
    """Builds and maintains BM25 index.

    Supports both full rebuilds and incremental updates.
    Uses stemming by default for better matching accuracy.
    """

    def __init__(self, db_path: str, use_stemming: bool = True):
        """Initialize the BM25 index.

        Args:
            db_path: Path to the SQLite database.
            use_stemming: Whether to apply Porter stemming during indexing.
        """
        self.db_path = db_path
        self.use_stemming = use_stemming

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
                tokens = tokenize_for_bm25(content, remove_stopwords=True,
                                           stem=self.use_stemming)
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

    def index_chunk(self, chunk_id: int, content: str) -> None:
        """Index a single chunk (incremental update).

        This is more efficient than rebuilding the entire index when
        adding new documents.

        Args:
            chunk_id: ID of the chunk to index.
            content: Content of the chunk.
        """
        conn = get_connection(self.db_path)

        try:
            # Tokenize and count terms
            tokens = tokenize_for_bm25(content, remove_stopwords=True,
                                       stem=self.use_stemming)
            term_freqs = Counter(tokens)

            # Get old terms for this chunk (if re-indexing)
            old_terms = get_doc_terms(conn, chunk_id)

            # Clear existing terms
            conn.execute("DELETE FROM doc_terms WHERE chunk_id = ?", (chunk_id,))

            # Insert new terms
            if term_freqs:
                insert_doc_terms(conn, chunk_id, dict(term_freqs), auto_commit=False)

            # Update term document frequencies incrementally
            terms_to_update: Dict[str, int] = {}

            # Decrement for old terms that aren't in new content
            for term in set(old_terms.keys()) - set(term_freqs.keys()):
                current = get_term_doc_frequency(conn, term)
                if current > 0:
                    terms_to_update[term] = current - 1

            # Increment for new terms that weren't in old content
            for term in set(term_freqs.keys()) - set(old_terms.keys()):
                current = get_term_doc_frequency(conn, term)
                terms_to_update[term] = current + 1

            if terms_to_update:
                update_term_doc_frequencies(conn, terms_to_update, auto_commit=False)

            # Update corpus stats
            stats = get_corpus_stats(conn)
            if stats:
                old_doc_length = sum(old_terms.values()) if old_terms else 0
                new_doc_length = len(tokens)

                # Adjust average document length
                total_docs = stats['total_docs']
                old_avg = stats['avg_doc_length']

                if old_terms:
                    # Re-indexing existing chunk
                    total_length = old_avg * total_docs - old_doc_length + new_doc_length
                else:
                    # New chunk
                    total_length = old_avg * total_docs + new_doc_length
                    total_docs += 1

                new_avg = total_length / total_docs if total_docs > 0 else 0
                update_corpus_stats(conn, total_docs, new_avg, auto_commit=False)

            conn.commit()
            logger.debug(f"Indexed chunk {chunk_id} with {len(term_freqs)} unique terms")

        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def remove_chunk(self, chunk_id: int) -> None:
        """Remove a chunk from the index.

        Args:
            chunk_id: ID of the chunk to remove.
        """
        conn = get_connection(self.db_path)

        try:
            # Get terms for this chunk
            old_terms = get_doc_terms(conn, chunk_id)

            if not old_terms:
                return  # Chunk not indexed

            # Remove terms
            conn.execute("DELETE FROM doc_terms WHERE chunk_id = ?", (chunk_id,))

            # Decrement document frequencies for all terms
            terms_to_update = {}
            for term in old_terms:
                current = get_term_doc_frequency(conn, term)
                if current > 0:
                    terms_to_update[term] = current - 1

            if terms_to_update:
                update_term_doc_frequencies(conn, terms_to_update, auto_commit=False)

            # Update corpus stats
            stats = get_corpus_stats(conn)
            if stats and stats['total_docs'] > 0:
                doc_length = sum(old_terms.values())
                total_docs = stats['total_docs'] - 1
                if total_docs > 0:
                    total_length = stats['avg_doc_length'] * stats['total_docs'] - doc_length
                    new_avg = total_length / total_docs
                else:
                    new_avg = 0
                update_corpus_stats(conn, total_docs, new_avg, auto_commit=False)

            conn.commit()
            logger.debug(f"Removed chunk {chunk_id} from BM25 index")

        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def index_chunks_batch(self, chunks: List[Tuple[int, str]]) -> None:
        """Index multiple chunks efficiently.

        Args:
            chunks: List of (chunk_id, content) tuples.
        """
        if not chunks:
            return

        conn = get_connection(self.db_path)

        try:
            term_doc_counts: Dict[str, int] = Counter()
            total_length = 0

            for chunk_id, content in chunks:
                tokens = tokenize_for_bm25(content, remove_stopwords=True,
                                           stem=self.use_stemming)
                term_freqs = Counter(tokens)

                # Clear existing terms
                conn.execute("DELETE FROM doc_terms WHERE chunk_id = ?", (chunk_id,))

                # Insert new terms
                if term_freqs:
                    insert_doc_terms(conn, chunk_id, dict(term_freqs), auto_commit=False)

                # Track document frequencies
                for term in set(tokens):
                    term_doc_counts[term] += 1

                total_length += len(tokens)

            # Get current corpus stats
            stats = get_corpus_stats(conn)
            if stats:
                # Update existing stats
                total_docs = stats['total_docs'] + len(chunks)
                old_total_length = stats['avg_doc_length'] * stats['total_docs']
                avg_length = (old_total_length + total_length) / total_docs
            else:
                # First batch
                total_docs = len(chunks)
                avg_length = total_length / total_docs if chunks else 0

            update_corpus_stats(conn, total_docs, avg_length, auto_commit=False)

            # Update term document frequencies
            existing_freqs = get_term_doc_frequencies_batch(conn, list(term_doc_counts.keys()))
            for term, count in term_doc_counts.items():
                term_doc_counts[term] = existing_freqs.get(term, 0) + count

            update_term_doc_frequencies(conn, dict(term_doc_counts), auto_commit=False)

            conn.commit()
            logger.info(f"Batch indexed {len(chunks)} chunks, {len(term_doc_counts)} unique terms")

        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


class BM25Search:
    """Performs BM25 search over indexed chunks."""

    def __init__(self, db_path: str,
                 query_expander: Optional['QueryExpander'] = None):
        """Initialize the BM25 searcher.

        Args:
            db_path: Path to the SQLite database.
            query_expander: Optional QueryExpander for query expansion.
        """
        self.db_path = db_path
        self._query_expander = query_expander

    def search(self, query: str, top_k: int = 10,
               chunk_type: str = 'small',
               expand_query: bool = False,
               stem: bool = True) -> List[Tuple[int, float]]:
        """Search for chunks matching the query.

        Uses batch queries for efficiency when many chunks match.
        When expand_query is enabled, searches all query variations and
        takes the max score per chunk.

        Args:
            query: Search query.
            top_k: Maximum number of results.
            chunk_type: Type of chunks to search ('small' or 'large').
            expand_query: Whether to expand query using QueryExpander.
            stem: Whether to apply stemming to query terms.

        Returns:
            List of (chunk_id, score) tuples sorted by score descending.
        """
        # Collect all query variations
        queries_to_search = [query]
        if expand_query and self._query_expander:
            queries_to_search = self._query_expander.expand(query)
            logger.debug(f"Expanded query to {len(queries_to_search)} variations")

        # Accumulate scores across all query variations (take max per chunk)
        chunk_scores: Dict[int, float] = {}

        conn = get_connection(self.db_path)

        try:
            # Get corpus statistics
            stats = get_corpus_stats(conn)
            if not stats:
                return []

            total_docs = stats['total_docs']
            avg_doc_length = stats['avg_doc_length']

            for q in queries_to_search:
                query_terms = tokenize_for_bm25(q, remove_stopwords=True, stem=stem)

                if not query_terms:
                    continue

                # Batch fetch document frequencies for all query terms
                doc_frequencies = get_term_doc_frequencies_batch(conn, query_terms)

                # Get all chunks with matching terms
                placeholders = ','.join(['?' for _ in query_terms])
                cursor = conn.execute(
                    f"""SELECT DISTINCT chunk_id FROM doc_terms
                        WHERE term IN ({placeholders})""",
                    query_terms
                )
                candidate_chunk_ids = [row[0] for row in cursor.fetchall()]

                if not candidate_chunk_ids:
                    continue

                # Batch fetch term frequencies for all candidate chunks
                all_term_freqs = get_doc_terms_batch(conn, candidate_chunk_ids)

                # Score each candidate
                for chunk_id in candidate_chunk_ids:
                    term_freqs = all_term_freqs.get(chunk_id, {})
                    doc_length = sum(term_freqs.values())

                    score = bm25_score(
                        query_terms, term_freqs, doc_length,
                        avg_doc_length, doc_frequencies, total_docs
                    )

                    if score > 0:
                        # Take max score across query variations
                        if chunk_id not in chunk_scores:
                            chunk_scores[chunk_id] = score
                        else:
                            chunk_scores[chunk_id] = max(chunk_scores[chunk_id], score)

            # Convert to sorted list
            results = [(chunk_id, score) for chunk_id, score in chunk_scores.items()]
            results.sort(key=lambda x: x[1], reverse=True)

            return results[:top_k]

        finally:
            conn.close()
