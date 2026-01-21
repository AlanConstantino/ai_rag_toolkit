"""Entity extractor module for the RAG system.

Extracts entities and relationships from text using LLM.
"""

from typing import Dict, List, Any, Optional

from rag_system.utils import get_logger, safe_json_loads

logger = get_logger(__name__)


EXTRACTION_PROMPT = """Analyze this documentation page and extract structured information.

TEXT:
{text}

Extract:
1. ENTITIES - Important nouns representing systems, services, configurations,
   concepts, or tools mentioned in this documentation.
2. RELATIONSHIPS - How these entities connect to each other.

Format your response as JSON:
{{
  "entities": [
    {{
      "name": "exact name as it appears",
      "type": "system|config|concept|process|tool",
      "description": "one sentence description"
    }}
  ],
  "relationships": [
    {{
      "source": "entity name",
      "target": "entity name",
      "type": "depends_on|configures|part_of|connects_to|triggers|reads_from|writes_to",
      "description": "brief description of the relationship"
    }}
  ]
}}

Return ONLY valid JSON, no other text."""


def validate_entity(entity: Dict[str, Any]) -> bool:
    """Validate an entity has required fields.

    Args:
        entity: Entity dict to validate.

    Returns:
        True if valid.
    """
    required = ['name', 'type', 'description']
    return all(k in entity and entity[k] for k in required)


def validate_relationship(relationship: Dict[str, Any]) -> bool:
    """Validate a relationship has required fields.

    Args:
        relationship: Relationship dict to validate.

    Returns:
        True if valid.
    """
    required = ['source', 'target', 'type', 'description']
    return all(k in relationship and relationship[k] for k in required)


class EntityExtractor:
    """Extracts entities and relationships from text using LLM."""

    def __init__(self, chat_client: Optional[Any] = None):
        """Initialize the extractor.

        Args:
            chat_client: Chat API client for LLM calls.
        """
        self.chat_client = chat_client

    def extract(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities and relationships from text.

        Args:
            text: Document text.

        Returns:
            Dict with 'entities' and 'relationships' lists.
        """
        if not self.chat_client:
            logger.warning("No chat client configured for entity extraction")
            return {'entities': [], 'relationships': []}

        try:
            prompt = EXTRACTION_PROMPT.format(text=text[:5000])  # Limit text length
            result = self.chat_client.complete_json(prompt)

            entities = result.get('entities', [])
            relationships = result.get('relationships', [])

            # Validate and filter
            entities = [e for e in entities if validate_entity(e)]
            relationships = [r for r in relationships if validate_relationship(r)]

            return {
                'entities': entities,
                'relationships': relationships
            }

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {'entities': [], 'relationships': []}
