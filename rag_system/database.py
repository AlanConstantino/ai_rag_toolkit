"""Database module for the RAG system.

Provides SQLite database initialization and helper functions for CRUD operations.
"""

import sqlite3
import json
from typing import Optional, List, Dict, Any


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
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

def init_db(db_path: str) -> None:
    """Initialize the database with the schema.

    Args:
        db_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def get_connection(db_path: str) -> sqlite3.Connection:
    """Get a database connection with foreign keys enabled.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A sqlite3 Connection object.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# =============================================================================
# Page Operations
# =============================================================================

def insert_page(conn: sqlite3.Connection, url: str, title: str,
                raw_html: str, parsed_text: str, content_hash: str,
                summary: Optional[str] = None) -> int:
    """Insert a page into the database.

    Args:
        conn: Database connection.
        url: Page URL.
        title: Page title.
        raw_html: Raw HTML content.
        parsed_text: Extracted text content.
        content_hash: Hash of the content for change detection.
        summary: Optional page summary.

    Returns:
        The ID of the inserted page.
    """
    cursor = conn.execute(
        """INSERT INTO pages (url, title, raw_html, parsed_text, content_hash, summary)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (url, title, raw_html, parsed_text, content_hash, summary)
    )
    conn.commit()
    return cursor.lastrowid


def get_page_by_url(conn: sqlite3.Connection, url: str) -> Optional[Dict[str, Any]]:
    """Get a page by its URL.

    Args:
        conn: Database connection.
        url: Page URL to find.

    Returns:
        Page data as a dict, or None if not found.
    """
    cursor = conn.execute("SELECT * FROM pages WHERE url = ?", (url,))
    row = cursor.fetchone()
    return dict(row) if row else None


def update_page_summary(conn: sqlite3.Connection, page_id: int, summary: str) -> None:
    """Update the summary for a page.

    Args:
        conn: Database connection.
        page_id: ID of the page to update.
        summary: New summary text.
    """
    conn.execute("UPDATE pages SET summary = ? WHERE id = ?", (summary, page_id))
    conn.commit()


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
                 embedding: Optional[List[float]] = None) -> int:
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

    Returns:
        The ID of the inserted chunk.
    """
    embedding_json = json.dumps(embedding) if embedding else None
    cursor = conn.execute(
        """INSERT INTO chunks (page_id, parent_chunk_id, chunk_type, chunk_index,
                              content, heading_path, embedding_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (page_id, parent_chunk_id, chunk_type, chunk_index, content,
         heading_path, embedding_json)
    )
    conn.commit()
    return cursor.lastrowid


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
                          embedding: List[float]) -> None:
    """Update the embedding for a chunk.

    Args:
        conn: Database connection.
        chunk_id: ID of the chunk.
        embedding: Embedding vector.
    """
    embedding_json = json.dumps(embedding)
    conn.execute("UPDATE chunks SET embedding_json = ? WHERE id = ?",
                (embedding_json, chunk_id))
    conn.commit()


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


def delete_chunks_by_page(conn: sqlite3.Connection, page_id: int) -> int:
    """Delete all chunks for a page.

    Used when re-indexing a page whose content has changed.

    Args:
        conn: Database connection.
        page_id: ID of the page whose chunks should be deleted.

    Returns:
        Number of chunks deleted.
    """
    cursor = conn.execute("DELETE FROM chunks WHERE page_id = ?", (page_id,))
    conn.commit()
    return cursor.rowcount


def delete_doc_terms_by_page(conn: sqlite3.Connection, page_id: int) -> int:
    """Delete all BM25 doc_terms entries for a page's chunks.

    Used when re-indexing a page whose content has changed.

    Args:
        conn: Database connection.
        page_id: ID of the page whose terms should be deleted.

    Returns:
        Number of term entries deleted.
    """
    cursor = conn.execute(
        "DELETE FROM doc_terms WHERE chunk_id IN (SELECT id FROM chunks WHERE page_id = ?)",
        (page_id,)
    )
    conn.commit()
    return cursor.rowcount


def update_page_content(conn: sqlite3.Connection, page_id: int,
                        title: str, raw_html: str, parsed_text: str,
                        content_hash: str) -> None:
    """Update a page's content fields.

    Used when re-indexing a page whose content has changed.

    Args:
        conn: Database connection.
        page_id: ID of the page to update.
        title: New page title.
        raw_html: New raw HTML content.
        parsed_text: New parsed text content.
        content_hash: New content hash.
    """
    conn.execute(
        "UPDATE pages SET title = ?, raw_html = ?, parsed_text = ?, content_hash = ?, crawled_at = CURRENT_TIMESTAMP WHERE id = ?",
        (title, raw_html, parsed_text, content_hash, page_id)
    )
    conn.commit()


# =============================================================================
# Entity Operations
# =============================================================================

def insert_entity(conn: sqlite3.Connection, name: str, entity_type: str,
                  description: str, page_id: Optional[int] = None,
                  normalized_name: Optional[str] = None) -> int:
    """Insert an entity into the database.

    Args:
        conn: Database connection.
        name: Entity name.
        entity_type: Type (system, config, concept, process, tool).
        description: Entity description.
        page_id: Optional ID of the source page.
        normalized_name: Optional normalized name for deduplication.

    Returns:
        The ID of the inserted entity.
    """
    if normalized_name is None:
        normalized_name = name.strip().lower()
    cursor = conn.execute(
        """INSERT INTO entities (name, type, description, page_id, normalized_name)
           VALUES (?, ?, ?, ?, ?)""",
        (name, entity_type, description, page_id, normalized_name)
    )
    conn.commit()
    return cursor.lastrowid


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
                       description: str) -> int:
    """Insert a relationship between entities.

    Args:
        conn: Database connection.
        source_entity_id: ID of the source entity.
        target_entity_id: ID of the target entity.
        relationship_type: Type of relationship.
        description: Relationship description.

    Returns:
        The ID of the inserted relationship.
    """
    cursor = conn.execute(
        """INSERT INTO relationships (source_entity_id, target_entity_id,
                                     type, description)
           VALUES (?, ?, ?, ?)""",
        (source_entity_id, target_entity_id, relationship_type, description)
    )
    conn.commit()
    return cursor.lastrowid


def link_chunk_to_entity(conn: sqlite3.Connection, chunk_id: int,
                         entity_id: int) -> None:
    """Create a link between a chunk and an entity.

    Args:
        conn: Database connection.
        chunk_id: ID of the chunk.
        entity_id: ID of the entity.
    """
    conn.execute(
        "INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id) VALUES (?, ?)",
        (chunk_id, entity_id)
    )
    conn.commit()


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
                     terms: Dict[str, int]) -> None:
    """Insert term frequencies for a chunk.

    Args:
        conn: Database connection.
        chunk_id: ID of the chunk.
        terms: Dict mapping terms to their frequencies.
    """
    for term, freq in terms.items():
        conn.execute(
            "INSERT INTO doc_terms (chunk_id, term, term_frequency) VALUES (?, ?, ?)",
            (chunk_id, term, freq)
        )
    conn.commit()


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
                        avg_doc_length: float) -> None:
    """Update corpus statistics.

    Args:
        conn: Database connection.
        total_docs: Total number of documents.
        avg_doc_length: Average document length.
    """
    # Delete existing stats and insert new
    conn.execute("DELETE FROM corpus_stats")
    conn.execute(
        "INSERT INTO corpus_stats (total_docs, avg_doc_length) VALUES (?, ?)",
        (total_docs, avg_doc_length)
    )
    conn.commit()


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
                                term_freqs: Dict[str, int]) -> None:
    """Update term document frequencies.

    Args:
        conn: Database connection.
        term_freqs: Dict mapping terms to document frequencies.
    """
    for term, freq in term_freqs.items():
        conn.execute(
            """INSERT OR REPLACE INTO term_doc_frequencies (term, doc_frequency)
               VALUES (?, ?)""",
            (term, freq)
        )
    conn.commit()


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

def set_global_summary(conn: sqlite3.Connection, summary: str) -> None:
    """Set or replace the global summary.

    Args:
        conn: Database connection.
        summary: Global summary text.
    """
    conn.execute("DELETE FROM global_summary")
    conn.execute("INSERT INTO global_summary (content) VALUES (?)", (summary,))
    conn.commit()


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
                  summary: str) -> int:
    """Insert a system entry.

    Args:
        conn: Database connection.
        name: System name.
        description: System description.
        summary: System summary.

    Returns:
        The ID of the inserted system.
    """
    cursor = conn.execute(
        "INSERT INTO systems (name, description, summary) VALUES (?, ?, ?)",
        (name, description, summary)
    )
    conn.commit()
    return cursor.lastrowid


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
                expanded_queries: List[str], embedding: List[float]) -> None:
    """Cache a query's processed information.

    Args:
        conn: Database connection.
        query_hash: Hash of the original query.
        query_type: Classified query type.
        expanded_queries: List of expanded query variations.
        embedding: Query embedding vector.
    """
    conn.execute(
        """INSERT OR REPLACE INTO query_cache
           (query_hash, query_type, expanded_queries, embedding_json)
           VALUES (?, ?, ?, ?)""",
        (query_hash, query_type, json.dumps(expanded_queries), json.dumps(embedding))
    )
    conn.commit()


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
              confidence_score: float, answer_generated: bool) -> int:
    """Log a query execution.

    Args:
        conn: Database connection.
        query: Original query text.
        query_type: Classified query type.
        expanded_queries: List of expanded queries.
        retrieved_chunk_ids: IDs of retrieved chunks.
        confidence_score: Confidence score.
        answer_generated: Whether an answer was generated.

    Returns:
        The ID of the log entry.
    """
    cursor = conn.execute(
        """INSERT INTO query_log
           (query, query_type, expanded_queries, retrieved_chunk_ids,
            confidence_score, answer_generated)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (query, query_type, json.dumps(expanded_queries),
         json.dumps(retrieved_chunk_ids), confidence_score, answer_generated)
    )
    conn.commit()
    return cursor.lastrowid


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
