"""Tests for the entity resolver module."""

import unittest
import tempfile
import os

from rag_system.database import init_db, get_connection


class TestEntityNormalization(unittest.TestCase):
    """Test entity name normalization."""

    def test_normalize_name_lowercases(self):
        """normalize_name should lowercase entity names."""
        from rag_system.knowledge_graph.entity_resolver import normalize_name

        self.assertEqual(normalize_name('Redis'), 'redis')
        self.assertEqual(normalize_name('KUBERNETES'), 'kubernetes')

    def test_normalize_name_strips_whitespace(self):
        """normalize_name should strip whitespace."""
        from rag_system.knowledge_graph.entity_resolver import normalize_name

        self.assertEqual(normalize_name('  Redis  '), 'redis')
        self.assertEqual(normalize_name('\nKafka\t'), 'kafka')

    def test_normalize_name_handles_special_chars(self):
        """normalize_name should handle special characters."""
        from rag_system.knowledge_graph.entity_resolver import normalize_name

        self.assertEqual(normalize_name('Node.js'), 'node.js')
        self.assertEqual(normalize_name('C++'), 'c++')


class TestEntityMerging(unittest.TestCase):
    """Test entity merging logic."""

    def test_find_similar_entities(self):
        """find_similar_entities should find matching entities."""
        from rag_system.knowledge_graph.entity_resolver import find_similar_entities

        existing = [
            {'id': 1, 'name': 'Redis', 'normalized_name': 'redis'},
            {'id': 2, 'name': 'PostgreSQL', 'normalized_name': 'postgresql'},
        ]

        matches = find_similar_entities('redis', existing)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['id'], 1)

    def test_find_similar_entities_no_match(self):
        """find_similar_entities should return empty for no match."""
        from rag_system.knowledge_graph.entity_resolver import find_similar_entities

        existing = [
            {'id': 1, 'name': 'Redis', 'normalized_name': 'redis'},
        ]

        matches = find_similar_entities('kafka', existing)
        self.assertEqual(len(matches), 0)


class TestEntityResolver(unittest.TestCase):
    """Test EntityResolver class."""

    def setUp(self):
        """Set up test database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_path)

    def test_resolve_new_entity(self):
        """resolve should create new entity if not exists."""
        from rag_system.knowledge_graph.entity_resolver import EntityResolver

        resolver = EntityResolver(self.temp_path)

        entity = {
            'name': 'Redis',
            'type': 'system',
            'description': 'In-memory cache'
        }

        entity_id = resolver.resolve(entity)
        self.assertIsNotNone(entity_id)
        self.assertIsInstance(entity_id, int)

    def test_resolve_existing_entity(self):
        """resolve should return existing entity id if found."""
        from rag_system.knowledge_graph.entity_resolver import EntityResolver

        resolver = EntityResolver(self.temp_path)

        entity = {
            'name': 'Redis',
            'type': 'system',
            'description': 'In-memory cache'
        }

        # First resolve creates
        entity_id1 = resolver.resolve(entity)

        # Second resolve with same name returns existing
        entity2 = {
            'name': 'redis',  # Different case
            'type': 'system',
            'description': 'Cache system'
        }
        entity_id2 = resolver.resolve(entity2)

        self.assertEqual(entity_id1, entity_id2)

    def test_resolve_and_link_to_chunk(self):
        """resolve_and_link should create entity and link to chunk."""
        from rag_system.knowledge_graph.entity_resolver import EntityResolver
        from rag_system.database import insert_page, insert_chunk

        conn = get_connection(self.temp_path)
        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='', content_hash='a')
        chunk_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                               chunk_index=0, content='Redis content',
                               heading_path='')
        conn.close()

        resolver = EntityResolver(self.temp_path)

        entity = {
            'name': 'Redis',
            'type': 'system',
            'description': 'Cache'
        }

        entity_id = resolver.resolve_and_link(entity, chunk_id)

        # Verify link exists
        conn = get_connection(self.temp_path)
        cursor = conn.execute(
            "SELECT * FROM chunk_entities WHERE chunk_id = ? AND entity_id = ?",
            (chunk_id, entity_id)
        )
        link = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(link)

    def test_resolve_relationship(self):
        """resolve_relationship should create relationship between entities."""
        from rag_system.knowledge_graph.entity_resolver import EntityResolver

        resolver = EntityResolver(self.temp_path)

        # Create source and target entities
        source = {'name': 'App', 'type': 'system', 'description': 'Application'}
        target = {'name': 'Redis', 'type': 'system', 'description': 'Cache'}

        source_id = resolver.resolve(source)
        target_id = resolver.resolve(target)

        relationship = {
            'source': 'App',
            'target': 'Redis',
            'type': 'depends_on',
            'description': 'Uses for caching'
        }

        rel_id = resolver.resolve_relationship(relationship)

        self.assertIsNotNone(rel_id)

        # Verify relationship exists
        conn = get_connection(self.temp_path)
        cursor = conn.execute(
            "SELECT * FROM relationships WHERE id = ?",
            (rel_id,)
        )
        rel = cursor.fetchone()
        conn.close()

        self.assertEqual(rel['source_entity_id'], source_id)
        self.assertEqual(rel['target_entity_id'], target_id)

    def test_get_all_entities(self):
        """get_all_entities should return all entities."""
        from rag_system.knowledge_graph.entity_resolver import EntityResolver

        resolver = EntityResolver(self.temp_path)

        resolver.resolve({'name': 'Redis', 'type': 'system', 'description': 'Cache'})
        resolver.resolve({'name': 'Kafka', 'type': 'system', 'description': 'Queue'})

        entities = resolver.get_all_entities()
        self.assertEqual(len(entities), 2)


class TestBatchResolve(unittest.TestCase):
    """Test batch entity resolution."""

    def setUp(self):
        """Set up test database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_path)

    def test_resolve_batch(self):
        """resolve_batch should resolve multiple entities."""
        from rag_system.knowledge_graph.entity_resolver import EntityResolver

        resolver = EntityResolver(self.temp_path)

        entities = [
            {'name': 'Redis', 'type': 'system', 'description': 'Cache'},
            {'name': 'Kafka', 'type': 'system', 'description': 'Queue'},
        ]

        entity_ids = resolver.resolve_batch(entities)

        self.assertEqual(len(entity_ids), 2)
        self.assertTrue(all(isinstance(id, int) for id in entity_ids))


if __name__ == '__main__':
    unittest.main()
