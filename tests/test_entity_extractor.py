"""Tests for the entity extractor module."""

import unittest
from unittest.mock import MagicMock, patch
import json


class TestEntityExtractor(unittest.TestCase):
    """Test entity extraction from text."""

    def test_extract_entities_returns_structured_data(self):
        """extract_entities should return entities and relationships."""
        from rag_system.knowledge_graph.entity_extractor import EntityExtractor

        mock_client = MagicMock()
        mock_client.complete_json.return_value = {
            'entities': [
                {'name': 'Redis', 'type': 'system', 'description': 'Cache'}
            ],
            'relationships': [
                {'source': 'App', 'target': 'Redis', 'type': 'depends_on', 'description': 'Uses for cache'}
            ]
        }

        extractor = EntityExtractor(chat_client=mock_client)
        result = extractor.extract('App uses Redis for caching.')

        self.assertIn('entities', result)
        self.assertIn('relationships', result)
        self.assertEqual(len(result['entities']), 1)
        self.assertEqual(result['entities'][0]['name'], 'Redis')

    def test_extract_entities_handles_empty_response(self):
        """extract_entities should handle empty responses."""
        from rag_system.knowledge_graph.entity_extractor import EntityExtractor

        mock_client = MagicMock()
        mock_client.complete_json.return_value = {'entities': [], 'relationships': []}

        extractor = EntityExtractor(chat_client=mock_client)
        result = extractor.extract('Simple text.')

        self.assertEqual(result['entities'], [])
        self.assertEqual(result['relationships'], [])

    def test_extract_entities_handles_api_error(self):
        """extract_entities should handle API errors gracefully."""
        from rag_system.knowledge_graph.entity_extractor import EntityExtractor

        mock_client = MagicMock()
        mock_client.complete_json.side_effect = Exception('API error')

        extractor = EntityExtractor(chat_client=mock_client)
        result = extractor.extract('Some text')

        self.assertEqual(result['entities'], [])
        self.assertEqual(result['relationships'], [])


class TestEntityValidation(unittest.TestCase):
    """Test entity validation."""

    def test_validate_entity(self):
        """validate_entity should check required fields."""
        from rag_system.knowledge_graph.entity_extractor import validate_entity

        valid = {'name': 'Redis', 'type': 'system', 'description': 'Cache'}
        self.assertTrue(validate_entity(valid))

        invalid = {'name': 'Redis'}  # Missing type
        self.assertFalse(validate_entity(invalid))

    def test_validate_relationship(self):
        """validate_relationship should check required fields."""
        from rag_system.knowledge_graph.entity_extractor import validate_relationship

        valid = {'source': 'A', 'target': 'B', 'type': 'depends_on', 'description': 'X'}
        self.assertTrue(validate_relationship(valid))

        invalid = {'source': 'A'}  # Missing target
        self.assertFalse(validate_relationship(invalid))


if __name__ == '__main__':
    unittest.main()
