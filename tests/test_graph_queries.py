"""Tests for the graph queries module."""

import unittest
import tempfile
import os

from rag_system.database import (
    init_db, get_connection, insert_page, insert_chunk,
    insert_entity, insert_relationship, link_chunk_to_entity
)


class TestGraphQueries(unittest.TestCase):
    """Test graph query functionality."""

    def setUp(self):
        """Set up test database with entities and relationships."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)

        # Create a page and chunks
        self.page_id = insert_page(conn, url='http://test.com', title='Test',
                                   raw_html='', parsed_text='', content_hash='a')
        self.chunk1_id = insert_chunk(conn, page_id=self.page_id, chunk_type='small',
                                      chunk_index=0, content='Redis content',
                                      heading_path='Redis')
        self.chunk2_id = insert_chunk(conn, page_id=self.page_id, chunk_type='small',
                                      chunk_index=1, content='Kafka content',
                                      heading_path='Kafka')

        # Create entities
        self.entity1_id = insert_entity(conn, name='Redis', entity_type='system',
                                        description='In-memory cache',
                                        normalized_name='redis')
        self.entity2_id = insert_entity(conn, name='Kafka', entity_type='system',
                                        description='Message queue',
                                        normalized_name='kafka')
        self.entity3_id = insert_entity(conn, name='App', entity_type='system',
                                        description='Application',
                                        normalized_name='app')

        # Create relationships
        self.rel1_id = insert_relationship(conn, source_entity_id=self.entity3_id,
                                           target_entity_id=self.entity1_id,
                                           relationship_type='depends_on',
                                           description='Uses for caching')
        self.rel2_id = insert_relationship(conn, source_entity_id=self.entity3_id,
                                           target_entity_id=self.entity2_id,
                                           relationship_type='depends_on',
                                           description='Uses for messaging')

        # Link chunks to entities
        link_chunk_to_entity(conn, self.chunk1_id, self.entity1_id)
        link_chunk_to_entity(conn, self.chunk2_id, self.entity2_id)

        conn.close()

    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_path)

    def test_get_entity_by_name(self):
        """get_entity_by_name should find entity by name."""
        from rag_system.knowledge_graph.graph_queries import GraphQueries

        gq = GraphQueries(self.temp_path)
        entity = gq.get_entity_by_name('Redis')

        self.assertIsNotNone(entity)
        self.assertEqual(entity['name'], 'Redis')

    def test_get_entity_by_name_case_insensitive(self):
        """get_entity_by_name should be case insensitive."""
        from rag_system.knowledge_graph.graph_queries import GraphQueries

        gq = GraphQueries(self.temp_path)
        entity = gq.get_entity_by_name('redis')

        self.assertIsNotNone(entity)
        self.assertEqual(entity['name'], 'Redis')

    def test_get_related_entities(self):
        """get_related_entities should find connected entities."""
        from rag_system.knowledge_graph.graph_queries import GraphQueries

        gq = GraphQueries(self.temp_path)
        related = gq.get_related_entities(self.entity3_id)

        # App depends on Redis and Kafka
        self.assertEqual(len(related), 2)
        entity_ids = [r['entity_id'] for r in related]
        self.assertIn(self.entity1_id, entity_ids)
        self.assertIn(self.entity2_id, entity_ids)

    def test_get_entity_chunks(self):
        """get_entity_chunks should return chunks linked to entity."""
        from rag_system.knowledge_graph.graph_queries import GraphQueries

        gq = GraphQueries(self.temp_path)
        chunks = gq.get_entity_chunks(self.entity1_id)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['id'], self.chunk1_id)

    def test_get_relationship_path(self):
        """get_relationship_path should find path between entities."""
        from rag_system.knowledge_graph.graph_queries import GraphQueries

        gq = GraphQueries(self.temp_path)
        path = gq.get_relationship_path(self.entity3_id, self.entity1_id)

        self.assertIsNotNone(path)
        self.assertEqual(len(path), 1)  # Direct relationship
        self.assertEqual(path[0]['type'], 'depends_on')

    def test_get_all_entities(self):
        """get_all_entities should return all entities."""
        from rag_system.knowledge_graph.graph_queries import GraphQueries

        gq = GraphQueries(self.temp_path)
        entities = gq.get_all_entities()

        self.assertEqual(len(entities), 3)

    def test_get_all_relationships(self):
        """get_all_relationships should return all relationships."""
        from rag_system.knowledge_graph.graph_queries import GraphQueries

        gq = GraphQueries(self.temp_path)
        relationships = gq.get_all_relationships()

        self.assertEqual(len(relationships), 2)

    def test_search_entities(self):
        """search_entities should find entities matching query."""
        from rag_system.knowledge_graph.graph_queries import GraphQueries

        gq = GraphQueries(self.temp_path)
        results = gq.search_entities('cache')

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['name'], 'Redis')

    def test_get_entity_subgraph(self):
        """get_entity_subgraph should return entity with relationships."""
        from rag_system.knowledge_graph.graph_queries import GraphQueries

        gq = GraphQueries(self.temp_path)
        subgraph = gq.get_entity_subgraph(self.entity3_id)

        self.assertIn('entity', subgraph)
        self.assertIn('outgoing', subgraph)
        self.assertIn('incoming', subgraph)
        self.assertEqual(len(subgraph['outgoing']), 2)
        self.assertEqual(len(subgraph['incoming']), 0)


class TestGraphBoost(unittest.TestCase):
    """Test graph-based boost calculation."""

    def setUp(self):
        """Set up test database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)

        page_id = insert_page(conn, url='http://test.com', title='Test',
                             raw_html='', parsed_text='', content_hash='a')
        self.chunk1_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=0, content='Redis content',
                                      heading_path='')
        self.chunk2_id = insert_chunk(conn, page_id=page_id, chunk_type='small',
                                      chunk_index=1, content='Other content',
                                      heading_path='')

        entity_id = insert_entity(conn, name='Redis', entity_type='system',
                                  description='Cache', normalized_name='redis')
        link_chunk_to_entity(conn, self.chunk1_id, entity_id)

        conn.close()

    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_path)

    def test_calculate_graph_boost(self):
        """calculate_graph_boost should boost chunks with matching entities."""
        from rag_system.knowledge_graph.graph_queries import GraphQueries

        gq = GraphQueries(self.temp_path)

        # Query mentions Redis
        boost = gq.calculate_graph_boost([self.chunk1_id, self.chunk2_id], 'redis cache')

        # Chunk1 should get a boost (has Redis entity)
        self.assertGreater(boost.get(self.chunk1_id, 1.0), 1.0)
        # Chunk2 should have no boost
        self.assertEqual(boost.get(self.chunk2_id, 1.0), 1.0)


class TestEmptyGraph(unittest.TestCase):
    """Test with empty graph."""

    def setUp(self):
        """Set up empty test database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_path)

    def test_search_entities_empty(self):
        """search_entities should return empty list for empty graph."""
        from rag_system.knowledge_graph.graph_queries import GraphQueries

        gq = GraphQueries(self.temp_path)
        results = gq.search_entities('redis')

        self.assertEqual(results, [])

    def test_get_all_entities_empty(self):
        """get_all_entities should return empty list for empty graph."""
        from rag_system.knowledge_graph.graph_queries import GraphQueries

        gq = GraphQueries(self.temp_path)
        entities = gq.get_all_entities()

        self.assertEqual(entities, [])


if __name__ == '__main__':
    unittest.main()
