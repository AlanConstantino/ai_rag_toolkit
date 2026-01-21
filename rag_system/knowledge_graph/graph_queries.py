"""Graph queries module for the RAG system.

Provides query functions for the knowledge graph.
"""

from typing import Dict, List, Any, Optional

from rag_system.database import get_connection
from rag_system.utils import get_logger, tokenize

logger = get_logger(__name__)


class GraphQueries:
    """Provides query functions for the knowledge graph."""

    ENTITY_MATCH_BOOST = 1.2  # Boost for entity mention matches

    def __init__(self, db_path: str):
        """Initialize graph queries.

        Args:
            db_path: Path to the SQLite database.
        """
        self.db_path = db_path

    def get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get entity by name (case-insensitive).

        Args:
            name: Entity name.

        Returns:
            Entity dict or None.
        """
        normalized = name.strip().lower()

        conn = get_connection(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT * FROM entities WHERE normalized_name = ?",
                (normalized,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_related_entities(self, entity_id: int) -> List[Dict[str, Any]]:
        """Get entities related to given entity.

        Args:
            entity_id: ID of the entity.

        Returns:
            List of related entity info with relationship details.
        """
        conn = get_connection(self.db_path)
        try:
            # Get outgoing relationships
            cursor = conn.execute(
                """SELECT r.*, e.name as target_name, e.type as target_type,
                          e.description as target_description,
                          r.target_entity_id as entity_id
                   FROM relationships r
                   JOIN entities e ON r.target_entity_id = e.id
                   WHERE r.source_entity_id = ?""",
                (entity_id,)
            )
            outgoing = [dict(row) for row in cursor]

            # Get incoming relationships
            cursor = conn.execute(
                """SELECT r.*, e.name as source_name, e.type as source_type,
                          e.description as source_description,
                          r.source_entity_id as entity_id
                   FROM relationships r
                   JOIN entities e ON r.source_entity_id = e.id
                   WHERE r.target_entity_id = ?""",
                (entity_id,)
            )
            incoming = [dict(row) for row in cursor]

            return outgoing + incoming
        finally:
            conn.close()

    def get_entity_chunks(self, entity_id: int) -> List[Dict[str, Any]]:
        """Get chunks linked to an entity.

        Args:
            entity_id: ID of the entity.

        Returns:
            List of chunk dicts.
        """
        conn = get_connection(self.db_path)
        try:
            cursor = conn.execute(
                """SELECT c.* FROM chunks c
                   JOIN chunk_entities ce ON c.id = ce.chunk_id
                   WHERE ce.entity_id = ?""",
                (entity_id,)
            )
            return [dict(row) for row in cursor]
        finally:
            conn.close()

    def get_relationship_path(self, source_id: int,
                              target_id: int) -> Optional[List[Dict[str, Any]]]:
        """Find direct relationship path between entities.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.

        Returns:
            List of relationship dicts in path, or None if no path.
        """
        conn = get_connection(self.db_path)
        try:
            # Check for direct relationship
            cursor = conn.execute(
                """SELECT * FROM relationships
                   WHERE source_entity_id = ? AND target_entity_id = ?""",
                (source_id, target_id)
            )
            row = cursor.fetchone()
            if row:
                return [dict(row)]

            # Check reverse direction
            cursor = conn.execute(
                """SELECT * FROM relationships
                   WHERE source_entity_id = ? AND target_entity_id = ?""",
                (target_id, source_id)
            )
            row = cursor.fetchone()
            if row:
                return [dict(row)]

            return None
        finally:
            conn.close()

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Get all entities.

        Returns:
            List of entity dicts.
        """
        conn = get_connection(self.db_path)
        try:
            cursor = conn.execute("SELECT * FROM entities")
            return [dict(row) for row in cursor]
        finally:
            conn.close()

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """Get all relationships.

        Returns:
            List of relationship dicts.
        """
        conn = get_connection(self.db_path)
        try:
            cursor = conn.execute("SELECT * FROM relationships")
            return [dict(row) for row in cursor]
        finally:
            conn.close()

    def search_entities(self, query: str) -> List[Dict[str, Any]]:
        """Search entities by name or description.

        Args:
            query: Search query.

        Returns:
            List of matching entity dicts.
        """
        query_lower = query.lower()
        query_terms = tokenize(query)

        conn = get_connection(self.db_path)
        try:
            cursor = conn.execute("SELECT * FROM entities")
            results = []

            for row in cursor:
                entity = dict(row)
                name_lower = (entity.get('name') or '').lower()
                desc_lower = (entity.get('description') or '').lower()

                # Check if any query term appears in name or description
                for term in query_terms:
                    if term in name_lower or term in desc_lower:
                        results.append(entity)
                        break

            return results
        finally:
            conn.close()

    def get_entity_subgraph(self, entity_id: int) -> Dict[str, Any]:
        """Get entity with its immediate relationships.

        Args:
            entity_id: ID of the entity.

        Returns:
            Dict with 'entity', 'outgoing', and 'incoming' keys.
        """
        conn = get_connection(self.db_path)
        try:
            # Get entity
            cursor = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
            entity_row = cursor.fetchone()
            entity = dict(entity_row) if entity_row else None

            # Get outgoing relationships
            cursor = conn.execute(
                """SELECT r.*, e.name as target_name
                   FROM relationships r
                   JOIN entities e ON r.target_entity_id = e.id
                   WHERE r.source_entity_id = ?""",
                (entity_id,)
            )
            outgoing = [dict(row) for row in cursor]

            # Get incoming relationships
            cursor = conn.execute(
                """SELECT r.*, e.name as source_name
                   FROM relationships r
                   JOIN entities e ON r.source_entity_id = e.id
                   WHERE r.target_entity_id = ?""",
                (entity_id,)
            )
            incoming = [dict(row) for row in cursor]

            return {
                'entity': entity,
                'outgoing': outgoing,
                'incoming': incoming
            }
        finally:
            conn.close()

    def calculate_graph_boost(self, chunk_ids: List[int],
                              query: str) -> Dict[int, float]:
        """Calculate boost factors based on entity matches.

        Args:
            chunk_ids: List of chunk IDs.
            query: Query text.

        Returns:
            Dict mapping chunk_id to boost factor.
        """
        query_terms = set(tokenize(query))
        if not query_terms:
            return {}

        conn = get_connection(self.db_path)
        try:
            # Get all entities
            cursor = conn.execute("SELECT id, normalized_name FROM entities")
            entity_names = {row['id']: row['normalized_name'] for row in cursor}

            # Find entities mentioned in query
            matching_entity_ids = set()
            for entity_id, normalized_name in entity_names.items():
                if normalized_name in query_terms:
                    matching_entity_ids.add(entity_id)

            if not matching_entity_ids:
                return {}

            # Get chunks linked to matching entities
            boost = {}
            placeholders = ','.join('?' * len(chunk_ids))
            cursor = conn.execute(
                f"""SELECT chunk_id, entity_id FROM chunk_entities
                    WHERE chunk_id IN ({placeholders})""",
                chunk_ids
            )

            for row in cursor:
                if row['entity_id'] in matching_entity_ids:
                    boost[row['chunk_id']] = self.ENTITY_MATCH_BOOST

            return boost
        finally:
            conn.close()
