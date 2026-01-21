"""Entity resolver module for the RAG system.

Resolves and deduplicates entities across documents.
"""

from typing import Dict, List, Any, Optional

from rag_system.database import get_connection
from rag_system.utils import get_logger

logger = get_logger(__name__)


def normalize_name(name: str) -> str:
    """Normalize entity name for matching.

    Args:
        name: Entity name.

    Returns:
        Normalized name.
    """
    return name.strip().lower()


def find_similar_entities(normalized_name: str,
                          existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find entities with matching normalized name.

    Args:
        normalized_name: Normalized name to match.
        existing: List of existing entities with 'normalized_name' field.

    Returns:
        List of matching entities.
    """
    return [e for e in existing if e.get('normalized_name') == normalized_name]


class EntityResolver:
    """Resolves and deduplicates entities using database."""

    def __init__(self, db_path: str):
        """Initialize the resolver.

        Args:
            db_path: Path to the SQLite database.
        """
        self.db_path = db_path
        self._entity_cache: Dict[str, int] = {}

    def _get_or_create_entity(self, name: str, entity_type: str,
                              description: str) -> int:
        """Get existing entity or create new one.

        Args:
            name: Entity name.
            entity_type: Entity type.
            description: Entity description.

        Returns:
            Entity ID.
        """
        normalized = normalize_name(name)

        # Check cache first
        if normalized in self._entity_cache:
            return self._entity_cache[normalized]

        conn = get_connection(self.db_path)
        try:
            # Check if entity exists
            cursor = conn.execute(
                "SELECT id FROM entities WHERE normalized_name = ?",
                (normalized,)
            )
            row = cursor.fetchone()

            if row:
                entity_id = row['id']
            else:
                # Create new entity
                cursor = conn.execute(
                    """INSERT INTO entities (name, type, description, normalized_name)
                       VALUES (?, ?, ?, ?)""",
                    (name, entity_type, description, normalized)
                )
                conn.commit()
                entity_id = cursor.lastrowid

            self._entity_cache[normalized] = entity_id
            return entity_id

        finally:
            conn.close()

    def resolve(self, entity: Dict[str, Any]) -> int:
        """Resolve an entity, creating if necessary.

        Args:
            entity: Entity dict with 'name', 'type', 'description'.

        Returns:
            Entity ID.
        """
        return self._get_or_create_entity(
            entity['name'],
            entity['type'],
            entity['description']
        )

    def resolve_and_link(self, entity: Dict[str, Any], chunk_id: int) -> int:
        """Resolve entity and link to chunk.

        Args:
            entity: Entity dict.
            chunk_id: Chunk ID to link to.

        Returns:
            Entity ID.
        """
        entity_id = self.resolve(entity)

        conn = get_connection(self.db_path)
        try:
            # Check if link exists
            cursor = conn.execute(
                """SELECT 1 FROM chunk_entities
                   WHERE chunk_id = ? AND entity_id = ?""",
                (chunk_id, entity_id)
            )
            if not cursor.fetchone():
                conn.execute(
                    """INSERT INTO chunk_entities (chunk_id, entity_id)
                       VALUES (?, ?)""",
                    (chunk_id, entity_id)
                )
                conn.commit()
        finally:
            conn.close()

        return entity_id

    def resolve_relationship(self, relationship: Dict[str, Any]) -> Optional[int]:
        """Resolve a relationship between entities.

        Args:
            relationship: Relationship dict with 'source', 'target', 'type', 'description'.

        Returns:
            Relationship ID or None if entities don't exist.
        """
        source_normalized = normalize_name(relationship['source'])
        target_normalized = normalize_name(relationship['target'])

        conn = get_connection(self.db_path)
        try:
            # Get source entity
            cursor = conn.execute(
                "SELECT id FROM entities WHERE normalized_name = ?",
                (source_normalized,)
            )
            source_row = cursor.fetchone()
            if not source_row:
                logger.warning(f"Source entity not found: {relationship['source']}")
                return None

            # Get target entity
            cursor = conn.execute(
                "SELECT id FROM entities WHERE normalized_name = ?",
                (target_normalized,)
            )
            target_row = cursor.fetchone()
            if not target_row:
                logger.warning(f"Target entity not found: {relationship['target']}")
                return None

            source_id = source_row['id']
            target_id = target_row['id']

            # Check if relationship exists
            cursor = conn.execute(
                """SELECT id FROM relationships
                   WHERE source_entity_id = ? AND target_entity_id = ? AND type = ?""",
                (source_id, target_id, relationship['type'])
            )
            row = cursor.fetchone()

            if row:
                return row['id']

            # Create new relationship
            cursor = conn.execute(
                """INSERT INTO relationships (source_entity_id, target_entity_id, type, description)
                   VALUES (?, ?, ?, ?)""",
                (source_id, target_id, relationship['type'], relationship['description'])
            )
            conn.commit()
            return cursor.lastrowid

        finally:
            conn.close()

    def resolve_batch(self, entities: List[Dict[str, Any]]) -> List[int]:
        """Resolve multiple entities.

        Args:
            entities: List of entity dicts.

        Returns:
            List of entity IDs.
        """
        return [self.resolve(entity) for entity in entities]

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Get all entities from database.

        Returns:
            List of entity dicts.
        """
        conn = get_connection(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT id, name, type, description, normalized_name FROM entities"
            )
            return [dict(row) for row in cursor]
        finally:
            conn.close()

    def clear_cache(self) -> None:
        """Clear the entity cache."""
        self._entity_cache = {}
