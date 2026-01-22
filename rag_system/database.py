"""Database module for the RAG system.

Provides SQLite database initialization and helper functions for CRUD operations.
Includes error handling, retry logic, and transaction management for production use.
"""

import sqlite3
import json
import time
import functools
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Generator, Callable, TypeVar

# Type variable for generic decorator
T = TypeVar('T')


# =============================================================================
# Custom Exceptions
# =============================================================================

class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class TransactionError(DatabaseError):
    """Raised when a transaction fails."""
    pass


class IntegrityConstraintError(DatabaseError):
    """Raised when an integrity constraint is violated (e.g., duplicate key)."""
    pass


class QueryError(DatabaseError):
    """Raised when a query execution fails."""
    pass


# =============================================================================
# Retry Decorator
# =============================================================================

def retry_on_error(
    max_retries: int = 3,
    retry_delay: float = 0.5,
    exponential_backoff: bool = True,
    retryable_errors: tuple = (sqlite3.OperationalError,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry database operations on transient errors.

    Args:
        max_retries: Maximum number of retry attempts.
        retry_delay: Initial delay between retries in seconds.
        exponential_backoff: If True, double delay after each retry.
        retryable_errors: Tuple of exception types that trigger retry.

    Returns:
        Decorated function with retry logic.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            delay = retry_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_errors as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay)
                        if exponential_backoff:
                            delay *= 2
                    else:
                        raise ConnectionError(
                            f"Database operation failed after {max_retries + 1} attempts: {e}"
                        ) from e

            # Should never reach here, but satisfy type checker
            raise ConnectionError(
                f"Database operation failed: {last_exception}"
            ) from last_exception

        return wrapper
    return decorator


# =============================================================================
# Schema Definition
# =============================================================================

SCHEMA = """
-- ============================================
-- CORE CONTENT
-- ============================================

CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    raw_html TEXT,
    parsed_text TEXT,
    summary TEXT,
    content_hash TEXT,
    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    page_id INTEGER NOT NULL,
    parent_chunk_id INTEGER,
    chunk_type TEXT,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    heading_path TEXT,
    embedding_json TEXT,
    FOREIGN KEY (page_id) REFERENCES pages(id),
    FOREIGN KEY (parent_chunk_id) REFERENCES chunks(id)
);

-- ============================================
-- KNOWLEDGE GRAPH
-- ============================================

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT,
    description TEXT,
    normalized_name TEXT,
    page_id INTEGER,
    FOREIGN KEY (page_id) REFERENCES pages(id)
);

CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY,
    source_entity_id INTEGER NOT NULL,
    target_entity_id INTEGER NOT NULL,
    type TEXT,
    description TEXT,
    FOREIGN KEY (source_entity_id) REFERENCES entities(id),
    FOREIGN KEY (target_entity_id) REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, entity_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);

-- ============================================
-- SUMMARIZATION HIERARCHY
-- ============================================

CREATE TABLE IF NOT EXISTS systems (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS page_systems (
    page_id INTEGER NOT NULL,
    system_id INTEGER NOT NULL,
    PRIMARY KEY (page_id, system_id),
    FOREIGN KEY (page_id) REFERENCES pages(id),
    FOREIGN KEY (system_id) REFERENCES systems(id)
);

CREATE TABLE IF NOT EXISTS global_summary (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- BM25 INDEX
-- ============================================

CREATE TABLE IF NOT EXISTS doc_terms (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL,
    term TEXT NOT NULL,
    term_frequency INTEGER NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);

CREATE TABLE IF NOT EXISTS corpus_stats (
    id INTEGER PRIMARY KEY,
    total_docs INTEGER,
    avg_doc_length REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS term_doc_frequencies (
    term TEXT PRIMARY KEY,
    doc_frequency INTEGER NOT NULL
);

-- ============================================
-- CACHING & LOGGING
-- ============================================

CREATE TABLE IF NOT EXISTS query_cache (
    query_hash TEXT PRIMARY KEY,
    query_type TEXT,
    expanded_queries TEXT,
    embedding_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY,
    query TEXT,
    query_type TEXT,
    expanded_queries TEXT,
    retrieved_chunk_ids TEXT,
    confidence_score REAL,
    answer_generated BOOLEAN,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Timing metrics (in seconds)
    total_time_ms REAL,
    embedding_time_ms REAL,
    search_time_ms REAL,
    rerank_time_ms REAL,
    generation_time_ms REAL,
    metrics_json TEXT
);

-- ============================================
-- INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_normalized ON entities(normalized_name);
CREATE INDEX IF NOT EXISTS idx_doc_terms_term ON doc_terms(term);
CREATE INDEX IF NOT EXISTS idx_doc_terms_chunk ON doc_terms(chunk_id);
"""


# =============================================================================
# Connection Management
# =============================================================================

@retry_on_error(max_retries=3, retry_delay=0.5)
def init_db(db_path: str) -> None:
    """Initialize the database with the schema.

    Uses context manager to ensure connection is properly closed.

    Args:
        db_path: Path to the SQLite database file.

    Raises:
        ConnectionError: If database initialization fails after retries.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(SCHEMA)
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to initialize database: {e}") from e


@retry_on_error(max_retries=3, retry_delay=0.5)
def get_connection(db_path: str) -> sqlite3.Connection:
    """Get a database connection with foreign keys enabled.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A sqlite3 Connection object.

    Raises:
        ConnectionError: If connection fails after retries.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    except sqlite3.Error as e:
        raise ConnectionError(f"Failed to connect to database: {e}") from e


@contextmanager
def managed_connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections that ensures cleanup.

    Automatically closes the connection when the context exits, regardless
    of whether an exception occurred.

    Args:
        db_path: Path to the SQLite database file.

    Yields:
        Database connection.

    Raises:
        ConnectionError: If connection fails.

    Example:
        with managed_connection(db_path) as conn:
            # Use connection
            pass  # Connection automatically closed
    """
    conn = None
    try:
        conn = get_connection(db_path)
        yield conn
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass  # Ignore errors during close


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database transactions.

    Provides atomic transaction semantics - commits on success, rolls back on error.

    Args:
        conn: Database connection to use for the transaction.

    Yields:
        The connection object for use within the transaction.

    Raises:
        TransactionError: If the transaction fails.

    Example:
        with transaction(conn) as tx:
            insert_page(tx, ...)
            insert_chunk(tx, ...)
        # Both operations commit together or neither does
    """
    try:
        yield conn
        conn.commit()
    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise IntegrityConstraintError(
            f"Integrity constraint violated: {e}"
        ) from e
    except sqlite3.Error as e:
        conn.rollback()
        raise TransactionError(f"Transaction failed: {e}") from e
    except Exception as e:
        conn.rollback()
        raise TransactionError(f"Transaction failed due to unexpected error: {e}") from e


# =============================================================================
# Page Operations
# =============================================================================

def insert_page(conn: sqlite3.Connection, url: str, title: str,
                raw_html: str, parsed_text: str, content_hash: str,
                summary: Optional[str] = None,
                auto_commit: bool = True) -> int:
    """Insert a page into the database.

    Args:
        conn: Database connection.
        url: Page URL.
        title: Page title.
        raw_html: Raw HTML content.
        parsed_text: Extracted text content.
        content_hash: Hash of the content for change detection.
        summary: Optional page summary.
        auto_commit: If True, commit after insert. Set False when using transaction().

    Returns:
        The ID of the inserted page.

    Raises:
        IntegrityConstraintError: If a page with the same URL already exists.
        QueryError: If the insert fails for other reasons.
    """
    try:
        cursor = conn.execute(
            """INSERT INTO pages (url, title, raw_html, parsed_text, content_hash, summary)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (url, title, raw_html, parsed_text, content_hash, summary)
        )
        if auto_commit:
            conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        if auto_commit:
            conn.rollback()
        raise IntegrityConstraintError(
            f"Page with URL '{url}' already exists: {e}"
        ) from e
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to insert page: {e}") from e


def get_page_by_url(conn: sqlite3.Connection, url: str) -> Optional[Dict[str, Any]]:
    """Get a page by its URL.

    Args:
        conn: Database connection.
        url: Page URL to find.

    Returns:
        Page data as a dict, or None if not found.

    Raises:
        QueryError: If the query fails.
    """
    try:
        cursor = conn.execute("SELECT * FROM pages WHERE url = ?", (url,))
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        raise QueryError(f"Failed to get page by URL: {e}") from e


def update_page_summary(conn: sqlite3.Connection, page_id: int, summary: str,
                        auto_commit: bool = True) -> None:
    """Update the summary for a page.

    Args:
        conn: Database connection.
        page_id: ID of the page to update.
        summary: New summary text.
        auto_commit: If True, commit after update.

    Raises:
        QueryError: If the update fails.
    """
    try:
        conn.execute("UPDATE pages SET summary = ? WHERE id = ?", (summary, page_id))
        if auto_commit:
            conn.commit()
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to update page summary: {e}") from e


def get_all_pages(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Get all pages from the database.

    Args:
        conn: Database connection.

    Returns:
        List of page dicts.
    """
    cursor = conn.execute("SELECT * FROM pages")
    return [dict(row) for row in cursor.fetchall()]


# =============================================================================
# Chunk Operations
# =============================================================================

def insert_chunk(conn: sqlite3.Connection, page_id: int, chunk_type: str,
                 chunk_index: int, content: str, heading_path: str,
                 parent_chunk_id: Optional[int] = None,
                 embedding: Optional[List[float]] = None,
                 auto_commit: bool = True) -> int:
    """Insert a chunk into the database.

    Args:
        conn: Database connection.
        page_id: ID of the parent page.
        chunk_type: 'large' or 'small'.
        chunk_index: Index of this chunk within the page.
        content: Chunk text content.
        heading_path: Heading hierarchy path (e.g., "Section > Subsection").
        parent_chunk_id: Optional ID of parent chunk (for small chunks).
        embedding: Optional embedding vector.
        auto_commit: If True, commit after insert.

    Returns:
        The ID of the inserted chunk.

    Raises:
        IntegrityConstraintError: If foreign key constraint is violated.
        QueryError: If the insert fails for other reasons.
    """
    try:
        embedding_json = json.dumps(embedding) if embedding else None
        cursor = conn.execute(
            """INSERT INTO chunks (page_id, parent_chunk_id, chunk_type, chunk_index,
                                  content, heading_path, embedding_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (page_id, parent_chunk_id, chunk_type, chunk_index, content,
             heading_path, embedding_json)
        )
        if auto_commit:
            conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        if auto_commit:
            conn.rollback()
        raise IntegrityConstraintError(
            f"Failed to insert chunk (invalid page_id or parent_chunk_id?): {e}"
        ) from e
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to insert chunk: {e}") from e


def get_chunks_by_page(conn: sqlite3.Connection, page_id: int) -> List[Dict[str, Any]]:
    """Get all chunks for a page.

    Args:
        conn: Database connection.
        page_id: ID of the page.

    Returns:
        List of chunk dicts.
    """
    cursor = conn.execute("SELECT * FROM chunks WHERE page_id = ?", (page_id,))
    return [dict(row) for row in cursor.fetchall()]


def get_chunk_by_id(conn: sqlite3.Connection, chunk_id: int) -> Optional[Dict[str, Any]]:
    """Get a chunk by its ID.

    Args:
        conn: Database connection.
        chunk_id: ID of the chunk.

    Returns:
        Chunk data as a dict, or None if not found.
    """
    cursor = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def update_chunk_embedding(conn: sqlite3.Connection, chunk_id: int,
                          embedding: List[float],
                          auto_commit: bool = True) -> None:
    """Update the embedding for a chunk.

    Args:
        conn: Database connection.
        chunk_id: ID of the chunk.
        embedding: Embedding vector.
        auto_commit: If True, commit after update.

    Raises:
        QueryError: If the update fails.
    """
    try:
        embedding_json = json.dumps(embedding)
        conn.execute("UPDATE chunks SET embedding_json = ? WHERE id = ?",
                    (embedding_json, chunk_id))
        if auto_commit:
            conn.commit()
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to update chunk embedding: {e}") from e


def get_all_chunks_with_embeddings(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Get all chunks that have embeddings.

    Args:
        conn: Database connection.

    Returns:
        List of chunk dicts with embeddings.
    """
    cursor = conn.execute(
        "SELECT * FROM chunks WHERE embedding_json IS NOT NULL"
    )
    return [dict(row) for row in cursor.fetchall()]


def delete_chunks_by_page(conn: sqlite3.Connection, page_id: int,
                         auto_commit: bool = True) -> int:
    """Delete all chunks for a page.

    Used when re-indexing a page whose content has changed.

    Args:
        conn: Database connection.
        page_id: ID of the page whose chunks should be deleted.
        auto_commit: If True, commit after delete.

    Returns:
        Number of chunks deleted.

    Raises:
        QueryError: If the delete fails.
    """
    try:
        cursor = conn.execute("DELETE FROM chunks WHERE page_id = ?", (page_id,))
        if auto_commit:
            conn.commit()
        return cursor.rowcount
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to delete chunks for page: {e}") from e


def delete_doc_terms_by_page(conn: sqlite3.Connection, page_id: int,
                             auto_commit: bool = True) -> int:
    """Delete all BM25 doc_terms entries for a page's chunks.

    Used when re-indexing a page whose content has changed.

    Args:
        conn: Database connection.
        page_id: ID of the page whose terms should be deleted.
        auto_commit: If True, commit after delete.

    Returns:
        Number of term entries deleted.

    Raises:
        QueryError: If the delete fails.
    """
    try:
        cursor = conn.execute(
            "DELETE FROM doc_terms WHERE chunk_id IN (SELECT id FROM chunks WHERE page_id = ?)",
            (page_id,)
        )
        if auto_commit:
            conn.commit()
        return cursor.rowcount
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to delete doc terms for page: {e}") from e


def update_page_content(conn: sqlite3.Connection, page_id: int,
                        title: str, raw_html: str, parsed_text: str,
                        content_hash: str, auto_commit: bool = True) -> None:
    """Update a page's content fields.

    Used when re-indexing a page whose content has changed.

    Args:
        conn: Database connection.
        page_id: ID of the page to update.
        title: New page title.
        raw_html: New raw HTML content.
        parsed_text: New parsed text content.
        content_hash: New content hash.
        auto_commit: If True, commit after update.

    Raises:
        QueryError: If the update fails.
    """
    try:
        conn.execute(
            "UPDATE pages SET title = ?, raw_html = ?, parsed_text = ?, content_hash = ?, crawled_at = CURRENT_TIMESTAMP WHERE id = ?",
            (title, raw_html, parsed_text, content_hash, page_id)
        )
        if auto_commit:
            conn.commit()
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to update page content: {e}") from e


# =============================================================================
# Entity Operations
# =============================================================================

def insert_entity(conn: sqlite3.Connection, name: str, entity_type: str,
                  description: str, page_id: Optional[int] = None,
                  normalized_name: Optional[str] = None,
                  auto_commit: bool = True) -> int:
    """Insert an entity into the database.

    Args:
        conn: Database connection.
        name: Entity name.
        entity_type: Type (system, config, concept, process, tool).
        description: Entity description.
        page_id: Optional ID of the source page.
        normalized_name: Optional normalized name for deduplication.
        auto_commit: If True, commit after insert.

    Returns:
        The ID of the inserted entity.

    Raises:
        IntegrityConstraintError: If foreign key constraint is violated.
        QueryError: If the insert fails.
    """
    try:
        if normalized_name is None:
            normalized_name = name.strip().lower()
        cursor = conn.execute(
            """INSERT INTO entities (name, type, description, page_id, normalized_name)
               VALUES (?, ?, ?, ?, ?)""",
            (name, entity_type, description, page_id, normalized_name)
        )
        if auto_commit:
            conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        if auto_commit:
            conn.rollback()
        raise IntegrityConstraintError(f"Failed to insert entity: {e}") from e
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to insert entity: {e}") from e


def get_entity_by_name(conn: sqlite3.Connection, name: str) -> Optional[Dict[str, Any]]:
    """Get an entity by name.

    Args:
        conn: Database connection.
        name: Entity name to find.

    Returns:
        Entity data as a dict, or None if not found.
    """
    cursor = conn.execute("SELECT * FROM entities WHERE name = ?", (name,))
    row = cursor.fetchone()
    return dict(row) if row else None


def insert_relationship(conn: sqlite3.Connection, source_entity_id: int,
                       target_entity_id: int, relationship_type: str,
                       description: str, auto_commit: bool = True) -> int:
    """Insert a relationship between entities.

    Args:
        conn: Database connection.
        source_entity_id: ID of the source entity.
        target_entity_id: ID of the target entity.
        relationship_type: Type of relationship.
        description: Relationship description.
        auto_commit: If True, commit after insert.

    Returns:
        The ID of the inserted relationship.

    Raises:
        IntegrityConstraintError: If entity IDs don't exist.
        QueryError: If the insert fails.
    """
    try:
        cursor = conn.execute(
            """INSERT INTO relationships (source_entity_id, target_entity_id,
                                         type, description)
               VALUES (?, ?, ?, ?)""",
            (source_entity_id, target_entity_id, relationship_type, description)
        )
        if auto_commit:
            conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        if auto_commit:
            conn.rollback()
        raise IntegrityConstraintError(f"Failed to insert relationship: {e}") from e
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to insert relationship: {e}") from e


def link_chunk_to_entity(conn: sqlite3.Connection, chunk_id: int,
                         entity_id: int, auto_commit: bool = True) -> None:
    """Create a link between a chunk and an entity.

    Args:
        conn: Database connection.
        chunk_id: ID of the chunk.
        entity_id: ID of the entity.
        auto_commit: If True, commit after insert.

    Raises:
        QueryError: If the link fails.
    """
    try:
        conn.execute(
            "INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id) VALUES (?, ?)",
            (chunk_id, entity_id)
        )
        if auto_commit:
            conn.commit()
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to link chunk to entity: {e}") from e


def get_entities_for_chunk(conn: sqlite3.Connection,
                           chunk_id: int) -> List[Dict[str, Any]]:
    """Get all entities linked to a chunk.

    Args:
        conn: Database connection.
        chunk_id: ID of the chunk.

    Returns:
        List of entity dicts.
    """
    cursor = conn.execute(
        """SELECT e.* FROM entities e
           JOIN chunk_entities ce ON e.id = ce.entity_id
           WHERE ce.chunk_id = ?""",
        (chunk_id,)
    )
    return [dict(row) for row in cursor.fetchall()]


# =============================================================================
# BM25 Operations
# =============================================================================

def insert_doc_terms(conn: sqlite3.Connection, chunk_id: int,
                     terms: Dict[str, int], auto_commit: bool = True) -> None:
    """Insert term frequencies for a chunk.

    Args:
        conn: Database connection.
        chunk_id: ID of the chunk.
        terms: Dict mapping terms to their frequencies.
        auto_commit: If True, commit after all inserts.

    Raises:
        IntegrityConstraintError: If chunk_id doesn't exist.
        QueryError: If the insert fails.
    """
    try:
        for term, freq in terms.items():
            conn.execute(
                "INSERT INTO doc_terms (chunk_id, term, term_frequency) VALUES (?, ?, ?)",
                (chunk_id, term, freq)
            )
        if auto_commit:
            conn.commit()
    except sqlite3.IntegrityError as e:
        if auto_commit:
            conn.rollback()
        raise IntegrityConstraintError(f"Failed to insert doc terms: {e}") from e
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to insert doc terms: {e}") from e


def get_doc_terms(conn: sqlite3.Connection, chunk_id: int) -> Dict[str, int]:
    """Get term frequencies for a chunk.

    Args:
        conn: Database connection.
        chunk_id: ID of the chunk.

    Returns:
        Dict mapping terms to their frequencies.
    """
    cursor = conn.execute(
        "SELECT term, term_frequency FROM doc_terms WHERE chunk_id = ?",
        (chunk_id,)
    )
    return {row['term']: row['term_frequency'] for row in cursor.fetchall()}


def update_corpus_stats(conn: sqlite3.Connection, total_docs: int,
                        avg_doc_length: float, auto_commit: bool = True) -> None:
    """Update corpus statistics.

    Args:
        conn: Database connection.
        total_docs: Total number of documents.
        avg_doc_length: Average document length.
        auto_commit: If True, commit after update.

    Raises:
        QueryError: If the update fails.
    """
    try:
        # Delete existing stats and insert new
        conn.execute("DELETE FROM corpus_stats")
        conn.execute(
            "INSERT INTO corpus_stats (total_docs, avg_doc_length) VALUES (?, ?)",
            (total_docs, avg_doc_length)
        )
        if auto_commit:
            conn.commit()
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to update corpus stats: {e}") from e


def get_corpus_stats(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    """Get corpus statistics.

    Args:
        conn: Database connection.

    Returns:
        Stats dict with total_docs and avg_doc_length, or None.
    """
    cursor = conn.execute("SELECT * FROM corpus_stats LIMIT 1")
    row = cursor.fetchone()
    return dict(row) if row else None


def update_term_doc_frequencies(conn: sqlite3.Connection,
                                term_freqs: Dict[str, int],
                                auto_commit: bool = True) -> None:
    """Update term document frequencies.

    Args:
        conn: Database connection.
        term_freqs: Dict mapping terms to document frequencies.
        auto_commit: If True, commit after all updates.

    Raises:
        QueryError: If the update fails.
    """
    try:
        for term, freq in term_freqs.items():
            conn.execute(
                """INSERT OR REPLACE INTO term_doc_frequencies (term, doc_frequency)
                   VALUES (?, ?)""",
                (term, freq)
            )
        if auto_commit:
            conn.commit()
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to update term doc frequencies: {e}") from e


def get_term_doc_frequency(conn: sqlite3.Connection, term: str) -> int:
    """Get document frequency for a term.

    Args:
        conn: Database connection.
        term: Term to look up.

    Returns:
        Document frequency, or 0 if not found.
    """
    cursor = conn.execute(
        "SELECT doc_frequency FROM term_doc_frequencies WHERE term = ?",
        (term,)
    )
    row = cursor.fetchone()
    return row['doc_frequency'] if row else 0


# =============================================================================
# Summary Operations
# =============================================================================

def set_global_summary(conn: sqlite3.Connection, summary: str,
                       auto_commit: bool = True) -> None:
    """Set or replace the global summary.

    Args:
        conn: Database connection.
        summary: Global summary text.
        auto_commit: If True, commit after update.

    Raises:
        QueryError: If the update fails.
    """
    try:
        conn.execute("DELETE FROM global_summary")
        conn.execute("INSERT INTO global_summary (content) VALUES (?)", (summary,))
        if auto_commit:
            conn.commit()
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to set global summary: {e}") from e


def get_global_summary(conn: sqlite3.Connection) -> Optional[str]:
    """Get the global summary.

    Args:
        conn: Database connection.

    Returns:
        Global summary text, or None if not set.
    """
    cursor = conn.execute("SELECT content FROM global_summary LIMIT 1")
    row = cursor.fetchone()
    return row['content'] if row else None


def insert_system(conn: sqlite3.Connection, name: str, description: str,
                  summary: str, auto_commit: bool = True) -> int:
    """Insert a system entry.

    Args:
        conn: Database connection.
        name: System name.
        description: System description.
        summary: System summary.
        auto_commit: If True, commit after insert.

    Returns:
        The ID of the inserted system.

    Raises:
        QueryError: If the insert fails.
    """
    try:
        cursor = conn.execute(
            "INSERT INTO systems (name, description, summary) VALUES (?, ?, ?)",
            (name, description, summary)
        )
        if auto_commit:
            conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to insert system: {e}") from e


def get_system_by_name(conn: sqlite3.Connection, name: str) -> Optional[Dict[str, Any]]:
    """Get a system by name.

    Args:
        conn: Database connection.
        name: System name to find.

    Returns:
        System data as a dict, or None if not found.
    """
    cursor = conn.execute("SELECT * FROM systems WHERE name = ?", (name,))
    row = cursor.fetchone()
    return dict(row) if row else None


# =============================================================================
# Query Cache Operations
# =============================================================================

def cache_query(conn: sqlite3.Connection, query_hash: str, query_type: str,
                expanded_queries: List[str], embedding: List[float],
                auto_commit: bool = True) -> None:
    """Cache a query's processed information.

    Args:
        conn: Database connection.
        query_hash: Hash of the original query.
        query_type: Classified query type.
        expanded_queries: List of expanded query variations.
        embedding: Query embedding vector.
        auto_commit: If True, commit after insert.

    Raises:
        QueryError: If the cache operation fails.
    """
    try:
        conn.execute(
            """INSERT OR REPLACE INTO query_cache
               (query_hash, query_type, expanded_queries, embedding_json)
               VALUES (?, ?, ?, ?)""",
            (query_hash, query_type, json.dumps(expanded_queries), json.dumps(embedding))
        )
        if auto_commit:
            conn.commit()
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to cache query: {e}") from e


def get_cached_query(conn: sqlite3.Connection,
                     query_hash: str) -> Optional[Dict[str, Any]]:
    """Get cached query information.

    Args:
        conn: Database connection.
        query_hash: Hash of the query to find.

    Returns:
        Cached query data, or None if not found.
    """
    cursor = conn.execute(
        "SELECT * FROM query_cache WHERE query_hash = ?", (query_hash,)
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def log_query(conn: sqlite3.Connection, query: str, query_type: str,
              expanded_queries: List[str], retrieved_chunk_ids: List[int],
              confidence_score: float, answer_generated: bool,
              auto_commit: bool = True,
              metrics: Optional[Dict[str, float]] = None) -> int:
    """Log a query execution with optional timing metrics.

    Args:
        conn: Database connection.
        query: Original query text.
        query_type: Classified query type.
        expanded_queries: List of expanded queries.
        retrieved_chunk_ids: IDs of retrieved chunks.
        confidence_score: Confidence score.
        answer_generated: Whether an answer was generated.
        auto_commit: If True, commit after insert.
        metrics: Optional dict of timing metrics (in seconds).
            Keys: total_time, embedding_time, search_time, rerank_time, generation_time

    Returns:
        The ID of the log entry.

    Raises:
        QueryError: If logging fails.
    """
    # Convert metrics from seconds to milliseconds for storage
    total_time_ms = None
    embedding_time_ms = None
    search_time_ms = None
    rerank_time_ms = None
    generation_time_ms = None
    metrics_json = None

    if metrics:
        total_time_ms = metrics.get('total_time', 0) * 1000 if metrics.get('total_time') else None
        embedding_time_ms = metrics.get('embedding_time', 0) * 1000 if metrics.get('embedding_time') else None
        search_time_ms = metrics.get('search_time', 0) * 1000 if metrics.get('search_time') else None
        rerank_time_ms = metrics.get('rerank_time', 0) * 1000 if metrics.get('rerank_time') else None
        generation_time_ms = metrics.get('generation_time', 0) * 1000 if metrics.get('generation_time') else None
        metrics_json = json.dumps(metrics)

    try:
        cursor = conn.execute(
            """INSERT INTO query_log
               (query, query_type, expanded_queries, retrieved_chunk_ids,
                confidence_score, answer_generated, total_time_ms, embedding_time_ms,
                search_time_ms, rerank_time_ms, generation_time_ms, metrics_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (query, query_type, json.dumps(expanded_queries),
             json.dumps(retrieved_chunk_ids), confidence_score, answer_generated,
             total_time_ms, embedding_time_ms, search_time_ms, rerank_time_ms,
             generation_time_ms, metrics_json)
        )
        if auto_commit:
            conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to log query: {e}") from e


def get_query_logs(conn: sqlite3.Connection,
                   limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent query logs.

    Args:
        conn: Database connection.
        limit: Maximum number of logs to return.

    Returns:
        List of query log dicts.
    """
    cursor = conn.execute(
        "SELECT * FROM query_log ORDER BY timestamp DESC LIMIT ?", (limit,)
    )
    return [dict(row) for row in cursor.fetchall()]


# =============================================================================
# Schema Migration
# =============================================================================

def migrate_query_log_timing(conn: sqlite3.Connection,
                              auto_commit: bool = True) -> bool:
    """Add timing columns to query_log table if they don't exist.

    This migration adds performance metrics columns to existing databases.

    Args:
        conn: Database connection.
        auto_commit: If True, commit after migration.

    Returns:
        True if migration was applied, False if already migrated.

    Raises:
        QueryError: If migration fails.
    """
    # Check if columns already exist
    cursor = conn.execute("PRAGMA table_info(query_log)")
    columns = {row['name'] for row in cursor.fetchall()}

    if 'total_time_ms' in columns:
        return False  # Already migrated

    migration_columns = [
        ('total_time_ms', 'REAL'),
        ('embedding_time_ms', 'REAL'),
        ('search_time_ms', 'REAL'),
        ('rerank_time_ms', 'REAL'),
        ('generation_time_ms', 'REAL'),
        ('metrics_json', 'TEXT'),
    ]

    try:
        for col_name, col_type in migration_columns:
            if col_name not in columns:
                conn.execute(
                    f"ALTER TABLE query_log ADD COLUMN {col_name} {col_type}"
                )
        if auto_commit:
            conn.commit()
        return True
    except sqlite3.Error as e:
        if auto_commit:
            conn.rollback()
        raise QueryError(f"Failed to migrate query_log table: {e}") from e


# =============================================================================
# Analytics Export
# =============================================================================

def get_query_analytics(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Get analytics data from query logs.

    Returns aggregated statistics about query performance.

    Args:
        conn: Database connection.

    Returns:
        Dict containing analytics data.
    """
    analytics: Dict[str, Any] = {
        'total_queries': 0,
        'queries_with_answers': 0,
        'avg_confidence': 0.0,
        'query_types': {},
        'timing': {
            'avg_total_ms': None,
            'avg_embedding_ms': None,
            'avg_search_ms': None,
            'avg_rerank_ms': None,
            'avg_generation_ms': None,
        }
    }

    # Total queries and answers generated
    cursor = conn.execute(
        "SELECT COUNT(*) as total, SUM(CASE WHEN answer_generated THEN 1 ELSE 0 END) as with_answers FROM query_log"
    )
    row = cursor.fetchone()
    if row:
        analytics['total_queries'] = row['total'] or 0
        analytics['queries_with_answers'] = row['with_answers'] or 0

    # Average confidence
    cursor = conn.execute(
        "SELECT AVG(confidence_score) as avg_conf FROM query_log WHERE confidence_score IS NOT NULL"
    )
    row = cursor.fetchone()
    if row and row['avg_conf'] is not None:
        analytics['avg_confidence'] = round(row['avg_conf'], 4)

    # Query type distribution
    cursor = conn.execute(
        "SELECT query_type, COUNT(*) as count FROM query_log WHERE query_type IS NOT NULL GROUP BY query_type"
    )
    for row in cursor.fetchall():
        analytics['query_types'][row['query_type']] = row['count']

    # Timing averages
    cursor = conn.execute("""
        SELECT
            AVG(total_time_ms) as avg_total,
            AVG(embedding_time_ms) as avg_embedding,
            AVG(search_time_ms) as avg_search,
            AVG(rerank_time_ms) as avg_rerank,
            AVG(generation_time_ms) as avg_generation
        FROM query_log
        WHERE total_time_ms IS NOT NULL
    """)
    row = cursor.fetchone()
    if row:
        analytics['timing']['avg_total_ms'] = round(row['avg_total'], 2) if row['avg_total'] else None
        analytics['timing']['avg_embedding_ms'] = round(row['avg_embedding'], 2) if row['avg_embedding'] else None
        analytics['timing']['avg_search_ms'] = round(row['avg_search'], 2) if row['avg_search'] else None
        analytics['timing']['avg_rerank_ms'] = round(row['avg_rerank'], 2) if row['avg_rerank'] else None
        analytics['timing']['avg_generation_ms'] = round(row['avg_generation'], 2) if row['avg_generation'] else None

    return analytics


def export_query_logs_csv(conn: sqlite3.Connection, filepath: str,
                          limit: Optional[int] = None) -> int:
    """Export query logs to a CSV file.

    Args:
        conn: Database connection.
        filepath: Path to output CSV file.
        limit: Optional maximum number of rows to export.

    Returns:
        Number of rows exported.

    Raises:
        QueryError: If export fails.
    """
    import csv

    try:
        query = "SELECT * FROM query_log ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"

        cursor = conn.execute(query)
        rows = cursor.fetchall()

        if not rows:
            return 0

        # Get column names
        column_names = [description[0] for description in cursor.description]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(column_names)
            for row in rows:
                writer.writerow([row[col] for col in column_names])

        return len(rows)
    except (sqlite3.Error, IOError) as e:
        raise QueryError(f"Failed to export query logs: {e}") from e
