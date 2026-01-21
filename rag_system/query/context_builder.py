"""Context builder module for the RAG system.

Builds context from retrieved chunks for answer generation.
"""

from typing import Dict, List, Any, Optional

from rag_system.database import get_connection, get_chunk_by_id
from rag_system.utils import get_logger

logger = get_logger(__name__)


def format_chunk(chunk: Dict[str, Any],
                 include_heading: bool = False,
                 include_source: bool = False) -> str:
    """Format a chunk for context.

    Args:
        chunk: Chunk dict with 'content' and optionally 'heading_path'.
        include_heading: Whether to include heading path.
        include_source: Whether to include source info.

    Returns:
        Formatted chunk string.
    """
    parts = []

    if include_heading and chunk.get('heading_path'):
        parts.append(f"[{chunk['heading_path']}]")

    if include_source and chunk.get('page_title'):
        source = chunk.get('page_title', '')
        if chunk.get('page_url'):
            source = f"{source} ({chunk['page_url']})"
        parts.append(f"Source: {source}")

    parts.append(chunk.get('content', ''))

    return '\n'.join(parts)


def order_chunks(chunks: List[Dict[str, Any]],
                 by: str = 'score') -> List[Dict[str, Any]]:
    """Order chunks by specified criteria.

    Args:
        chunks: List of chunk dicts.
        by: Ordering criteria ('score' or 'position').

    Returns:
        Ordered list of chunks.
    """
    if by == 'score':
        return sorted(chunks, key=lambda c: c.get('score', 0), reverse=True)
    elif by == 'position':
        return sorted(chunks, key=lambda c: c.get('chunk_index', 0))
    else:
        return chunks


class ContextBuilder:
    """Builds context for answer generation."""

    def __init__(self, db_path: Optional[str] = None,
                 max_context_length: int = 4000):
        """Initialize the context builder.

        Args:
            db_path: Path to the SQLite database.
            max_context_length: Maximum context length in characters.
        """
        self.db_path = db_path
        self.max_context_length = max_context_length

    def enrich_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich chunks with page metadata.

        Args:
            chunks: List of chunk dicts with 'id'.

        Returns:
            Enriched chunk list.
        """
        if not self.db_path:
            return chunks

        conn = get_connection(self.db_path)
        try:
            enriched = []
            for chunk in chunks:
                chunk_id = chunk.get('id')
                if chunk_id:
                    # Get chunk details
                    db_chunk = get_chunk_by_id(conn, chunk_id)
                    if db_chunk:
                        # Merge database data
                        enriched_chunk = {**chunk, **dict(db_chunk)}

                        # Get page details
                        cursor = conn.execute(
                            "SELECT title, url FROM pages WHERE id = ?",
                            (db_chunk['page_id'],)
                        )
                        page = cursor.fetchone()
                        if page:
                            enriched_chunk['page_title'] = page['title']
                            enriched_chunk['page_url'] = page['url']

                        enriched.append(enriched_chunk)
                    else:
                        enriched.append(chunk)
                else:
                    enriched.append(chunk)

            return enriched
        finally:
            conn.close()

    def build_context(self, chunks: List[Dict[str, Any]],
                      include_headings: bool = False,
                      include_sources: bool = False,
                      order_by: str = 'score') -> str:
        """Build context string from chunks.

        Args:
            chunks: List of chunk dicts.
            include_headings: Whether to include heading paths.
            include_sources: Whether to include source info.
            order_by: Ordering criteria.

        Returns:
            Combined context string.
        """
        if not chunks:
            return ""

        # Enrich chunks with database info
        enriched = self.enrich_chunks(chunks)

        # Order chunks
        ordered = order_chunks(enriched, by=order_by)

        # Build context
        context_parts = []
        total_length = 0

        for i, chunk in enumerate(ordered):
            formatted = format_chunk(
                chunk,
                include_heading=include_headings,
                include_source=include_sources
            )

            # Check length limit
            if total_length + len(formatted) > self.max_context_length:
                # Truncate if needed
                remaining = self.max_context_length - total_length
                if remaining > 100:
                    formatted = formatted[:remaining] + "..."
                    context_parts.append(formatted)
                break

            context_parts.append(formatted)
            total_length += len(formatted)

        separator = "\n\n---\n\n"
        return separator.join(context_parts)

    def build_prompt_context(self, query: str,
                              chunks: List[Dict[str, Any]]) -> str:
        """Build context formatted for prompts.

        Args:
            query: Original query.
            chunks: List of chunk dicts.

        Returns:
            Prompt-ready context string.
        """
        context = self.build_context(
            chunks,
            include_headings=True,
            include_sources=True
        )

        return f"""Based on the following documentation context, answer the question.

CONTEXT:
{context}

QUESTION: {query}

Provide a clear, accurate answer based only on the context provided."""
