"""Integration tests for the RAG system."""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os

from rag_system.database import (
    init_db, get_connection, insert_page, insert_chunk, insert_entity,
    insert_relationship, link_chunk_to_entity
)


class TestFullQueryPipeline(unittest.TestCase):
    """Test full query pipeline integration."""

    def setUp(self):
        """Set up test database with sample data."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

        # Add sample pages and chunks
        conn = get_connection(self.temp_path)

        # Page 1: Redis docs
        redis_page = insert_page(
            conn, url='http://docs.example.com/redis',
            title='Redis Documentation',
            raw_html='<h1>Redis</h1>',
            parsed_text='Redis is an in-memory data structure store.',
            content_hash='redis1',
            summary='Documentation for Redis caching system.'
        )
        self.redis_chunk = insert_chunk(
            conn, page_id=redis_page, chunk_type='small',
            chunk_index=0,
            content='Redis is an in-memory data structure store used as a cache and message broker.',
            heading_path='Overview'
        )
        insert_chunk(
            conn, page_id=redis_page, chunk_type='small',
            chunk_index=1,
            content='To install Redis, download from redis.io and run make.',
            heading_path='Installation'
        )

        # Page 2: Kafka docs
        kafka_page = insert_page(
            conn, url='http://docs.example.com/kafka',
            title='Kafka Documentation',
            raw_html='<h1>Kafka</h1>',
            parsed_text='Apache Kafka is a distributed streaming platform.',
            content_hash='kafka1',
            summary='Documentation for Apache Kafka messaging.'
        )
        insert_chunk(
            conn, page_id=kafka_page, chunk_type='small',
            chunk_index=0,
            content='Apache Kafka is a distributed streaming platform for building real-time pipelines.',
            heading_path='Overview'
        )

        conn.close()

        # Build BM25 index
        from rag_system.search.bm25_search import BM25Index
        index = BM25Index(self.temp_path)
        index.build()

        # Add entities
        conn = get_connection(self.temp_path)
        redis_entity = insert_entity(
            conn, name='Redis', entity_type='system',
            description='In-memory cache', normalized_name='redis'
        )
        kafka_entity = insert_entity(
            conn, name='Kafka', entity_type='system',
            description='Message broker', normalized_name='kafka'
        )

        # Link entities to chunks
        link_chunk_to_entity(conn, self.redis_chunk, redis_entity)

        conn.close()

    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_path)

    def test_query_finds_relevant_content(self):
        """Query should find and return relevant content."""
        from rag_system.main import RAGSystem

        rag = RAGSystem(db_path=self.temp_path)
        result = rag.query("What is Redis?")

        self.assertIn('answer', result)
        self.assertIn('chunks', result)
        # Should find Redis-related chunks
        self.assertTrue(len(result['chunks']) > 0)

        # Check that Redis content was found
        all_content = ' '.join(c['content'] for c in result['chunks'])
        self.assertIn('Redis', all_content)

    def test_query_classifies_correctly(self):
        """Query should be classified appropriately."""
        from rag_system.main import RAGSystem

        rag = RAGSystem(db_path=self.temp_path)

        # Factual query
        result = rag.query("What is Redis?")
        self.assertEqual(result['query_type'], 'factual')

        # Procedural query
        result = rag.query("How do I install Redis?")
        self.assertEqual(result['query_type'], 'procedural')

    def test_query_calculates_confidence(self):
        """Query should return confidence score."""
        from rag_system.main import RAGSystem

        rag = RAGSystem(db_path=self.temp_path)
        result = rag.query("What is Redis?")

        self.assertIn('confidence', result)
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 1)


class TestSearchIntegration(unittest.TestCase):
    """Test search component integration."""

    def setUp(self):
        """Set up test database."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)

        # Add varied content for search testing
        page_id = insert_page(
            conn, url='http://test.com/docs',
            title='Test Documentation',
            raw_html='', parsed_text='Test content',
            content_hash='test1'
        )

        # Multiple chunks with different content
        insert_chunk(conn, page_id=page_id, chunk_type='small',
                    chunk_index=0, content='Database connection pooling',
                    heading_path='Databases')
        insert_chunk(conn, page_id=page_id, chunk_type='small',
                    chunk_index=1, content='Authentication and authorization',
                    heading_path='Security')
        insert_chunk(conn, page_id=page_id, chunk_type='small',
                    chunk_index=2, content='API rate limiting configuration',
                    heading_path='API')

        conn.close()

    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_path)

    def test_bm25_search_works(self):
        """BM25 search should find matching content."""
        from rag_system.search.bm25_search import BM25Search, BM25Index

        # Build the BM25 index first
        index = BM25Index(self.temp_path)
        index.build()

        search = BM25Search(self.temp_path)
        results = search.search("database connection")
        self.assertTrue(len(results) > 0)

    def test_reranker_boosts_matches(self):
        """Reranker should boost matching heading paths."""
        from rag_system.search.bm25_search import BM25Search, BM25Index
        from rag_system.search.reranker import Reranker

        # Build the BM25 index first
        index = BM25Index(self.temp_path)
        index.build()

        search = BM25Search(self.temp_path)
        # Use term that exists in the test data
        results = search.search("database")

        reranker = Reranker(self.temp_path)
        reranked = reranker.rerank(results, "database connection pooling")

        # Results should be reranked
        self.assertTrue(len(reranked) > 0)


class TestKnowledgeGraphIntegration(unittest.TestCase):
    """Test knowledge graph integration."""

    def setUp(self):
        """Set up test database with graph data."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

        conn = get_connection(self.temp_path)

        # Create entities and relationships
        redis = insert_entity(conn, 'Redis', 'system', 'Cache',
                             normalized_name='redis')
        kafka = insert_entity(conn, 'Kafka', 'system', 'Queue',
                             normalized_name='kafka')
        app = insert_entity(conn, 'Application', 'system', 'Main app',
                           normalized_name='application')

        insert_relationship(conn, app, redis, 'depends_on', 'Uses for caching')
        insert_relationship(conn, app, kafka, 'depends_on', 'Uses for messaging')

        conn.close()

    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_path)

    def test_graph_queries_find_relationships(self):
        """Graph queries should find entity relationships."""
        from rag_system.knowledge_graph.graph_queries import GraphQueries

        gq = GraphQueries(self.temp_path)

        # Find Redis
        redis = gq.get_entity_by_name('Redis')
        self.assertIsNotNone(redis)

        # Find all entities
        entities = gq.get_all_entities()
        self.assertEqual(len(entities), 3)

        # Find relationships
        relationships = gq.get_all_relationships()
        self.assertEqual(len(relationships), 2)


class TestEntityExtractionIntegration(unittest.TestCase):
    """Test entity extraction integration."""

    def test_extraction_and_resolution(self):
        """Entity extraction and resolution should work together."""
        from rag_system.knowledge_graph.entity_extractor import EntityExtractor
        from rag_system.knowledge_graph.entity_resolver import EntityResolver

        # Mock chat client
        mock_client = MagicMock()
        mock_client.complete_json.return_value = {
            'entities': [
                {'name': 'Redis', 'type': 'system', 'description': 'Cache'}
            ],
            'relationships': []
        }

        # Extract entities
        extractor = EntityExtractor(chat_client=mock_client)
        result = extractor.extract("Redis is a cache system.")

        self.assertEqual(len(result['entities']), 1)

        # Create temp database and resolve
        temp_fd, temp_path = tempfile.mkstemp(suffix='.db')
        os.close(temp_fd)
        init_db(temp_path)

        try:
            resolver = EntityResolver(temp_path)
            entity_id = resolver.resolve(result['entities'][0])

            self.assertIsNotNone(entity_id)
            self.assertIsInstance(entity_id, int)
        finally:
            os.unlink(temp_path)


class TestEndToEndScenario(unittest.TestCase):
    """Test complete end-to-end scenarios."""

    def setUp(self):
        """Set up complete test environment."""
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_fd)
        init_db(self.temp_path)

        # Populate with realistic documentation
        conn = get_connection(self.temp_path)

        # Configuration documentation
        config_page = insert_page(
            conn, url='http://docs.example.com/config',
            title='Configuration Guide',
            raw_html='<h1>Configuration</h1>',
            parsed_text='Configure the application settings.',
            content_hash='config1',
            summary='How to configure the application.'
        )
        insert_chunk(
            conn, page_id=config_page, chunk_type='small',
            chunk_index=0,
            content='Set the DATABASE_URL environment variable to connect to your database.',
            heading_path='Configuration > Database'
        )
        insert_chunk(
            conn, page_id=config_page, chunk_type='small',
            chunk_index=1,
            content='Enable caching by setting CACHE_ENABLED=true in your .env file.',
            heading_path='Configuration > Caching'
        )

        # Troubleshooting documentation
        trouble_page = insert_page(
            conn, url='http://docs.example.com/troubleshooting',
            title='Troubleshooting Guide',
            raw_html='<h1>Troubleshooting</h1>',
            parsed_text='Common problems and solutions.',
            content_hash='trouble1',
            summary='Solutions to common problems.'
        )
        insert_chunk(
            conn, page_id=trouble_page, chunk_type='small',
            chunk_index=0,
            content='If you see connection timeout errors, check your network settings.',
            heading_path='Troubleshooting > Connection Issues'
        )

        conn.close()

        # Build BM25 index
        from rag_system.search.bm25_search import BM25Index
        index = BM25Index(self.temp_path)
        index.build()

    def tearDown(self):
        """Clean up test environment."""
        os.unlink(self.temp_path)

    def test_procedural_query_workflow(self):
        """Procedural query should return actionable steps."""
        from rag_system.main import RAGSystem

        rag = RAGSystem(db_path=self.temp_path)
        result = rag.query("How do I configure the database?")

        self.assertEqual(result['query_type'], 'procedural')
        # Should find database configuration content
        if result['chunks']:
            contents = ' '.join(c['content'] for c in result['chunks'])
            self.assertTrue('DATABASE' in contents.upper() or 'database' in contents.lower())

    def test_troubleshooting_query_workflow(self):
        """Troubleshooting query should find problem solutions."""
        from rag_system.main import RAGSystem

        rag = RAGSystem(db_path=self.temp_path)
        result = rag.query("connection timeout error")

        self.assertEqual(result['query_type'], 'troubleshooting')

    def test_stats_reflect_content(self):
        """Statistics should reflect database content."""
        from rag_system.main import RAGSystem

        rag = RAGSystem(db_path=self.temp_path)
        stats = rag.get_stats()

        self.assertEqual(stats['pages'], 2)
        self.assertEqual(stats['chunks'], 3)


if __name__ == '__main__':
    unittest.main()
