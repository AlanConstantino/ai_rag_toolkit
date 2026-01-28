"""Context builder module for the RAG system.

Builds context from retrieved chunks for answer generation, including
knowledge graph context when available.
"""

from typing import Dict, List, Any, Optional, Set

from rag_system.database import get_connection, get_chunk_by_id, get_entities_for_chunk
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

    def get_entity_context(self, chunk_ids: List[int]) -> Dict[str, Any]:
        """Get entities and relationships related to chunks.

        Args:
            chunk_ids: List of chunk IDs to get entities for.

        Returns:
            Dict with 'entities' list and 'relationships' list.
        """
        if not self.db_path or not chunk_ids:
            return {'entities': [], 'relationships': []}

        conn = get_connection(self.db_path)
        try:
            # Collect unique entities from all chunks
            entities: Dict[int, Dict[str, Any]] = {}
            entity_ids: Set[int] = set()

            for chunk_id in chunk_ids:
                chunk_entities = get_entities_for_chunk(conn, chunk_id)
                for entity in chunk_entities:
                    entity_id = entity['id']
                    if entity_id not in entities:
                        entities[entity_id] = entity
                        entity_ids.add(entity_id)

            # Get relationships between collected entities
            relationships = []
            if entity_ids:
                placeholders = ','.join('?' * len(entity_ids))
                cursor = conn.execute(
                    f"""SELECT r.*, 
                               e1.name as source_name, 
                               e2.name as target_name
                        FROM relationships r
                        JOIN entities e1 ON r.source_entity_id = e1.id
                        JOIN entities e2 ON r.target_entity_id = e2.id
                        WHERE r.source_entity_id IN ({placeholders})
                          AND r.target_entity_id IN ({placeholders})""",
                    tuple(entity_ids) + tuple(entity_ids)
                )
                for row in cursor.fetchall():
                    relationships.append({
                        'source': row['source_name'],
                        'target': row['target_name'],
                        'type': row['relationship_type'],
                        'description': row['description']
                    })

            return {
                'entities': list(entities.values()),
                'relationships': relationships
            }
        finally:
            conn.close()

    def format_entity_context(self, entity_context: Dict[str, Any]) -> str:
        """Format entity context for inclusion in prompts.

        Args:
            entity_context: Dict with 'entities' and 'relationships'.

        Returns:
            Formatted string describing entities and their relationships.
        """
        parts = []

        entities = entity_context.get('entities', [])
        relationships = entity_context.get('relationships', [])

        if entities:
            parts.append("RELATED CONCEPTS:")
            for entity in entities[:10]:  # Limit to top 10 entities
                name = entity.get('name', '')
                entity_type = entity.get('type', 'concept')
                description = entity.get('description', '')
                if description:
                    parts.append(f"- {name} ({entity_type}): {description}")
                else:
                    parts.append(f"- {name} ({entity_type})")

        if relationships:
            parts.append("\nRELATIONSHIPS:")
            for rel in relationships[:10]:  # Limit to top 10 relationships
                source = rel.get('source', '')
                target = rel.get('target', '')
                rel_type = rel.get('type', 'related_to')
                parts.append(f"- {source} {rel_type.replace('_', ' ')} {target}")

        return '\n'.join(parts)

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

    def build_context_with_knowledge_graph(self, chunks: List[Dict[str, Any]],
                                            include_headings: bool = True,
                                            include_sources: bool = True,
                                            include_entities: bool = True) -> str:
        """Build context string including knowledge graph information.

        Args:
            chunks: List of chunk dicts with 'id'.
            include_headings: Whether to include heading paths.
            include_sources: Whether to include source info.
            include_entities: Whether to include entity context.

        Returns:
            Combined context string with optional entity information.
        """
        # Build base context
        base_context = self.build_context(
            chunks,
            include_headings=include_headings,
            include_sources=include_sources
        )

        if not include_entities:
            return base_context

        # Get entity context
        chunk_ids = [c.get('id') for c in chunks if c.get('id')]
        entity_context = self.get_entity_context(chunk_ids)

        # Format and append entity context if available
        if entity_context.get('entities') or entity_context.get('relationships'):
            entity_str = self.format_entity_context(entity_context)
            return f"{base_context}\n\n---\n\n{entity_str}"

        return base_context

    def build_rich_prompt_context(self, query: str,
                                   chunks: List[Dict[str, Any]],
                                   include_entities: bool = True) -> str:
        """Build context formatted for prompts with knowledge graph info.

        Args:
            query: Original query.
            chunks: List of chunk dicts.
            include_entities: Whether to include entity context.

        Returns:
            Prompt-ready context string with optional entity information.
        """
        context = self.build_context_with_knowledge_graph(
            chunks,
            include_headings=True,
            include_sources=True,
            include_entities=include_entities
        )

        return f"""Based on the following documentation context, answer the question.

CONTEXT:
{context}

QUESTION: {query}

Provide a clear, accurate answer based only on the context provided."""
