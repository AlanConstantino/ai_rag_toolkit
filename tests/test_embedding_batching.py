"""Tests for embedding batch rate limiting functionality.

Tests the ability to process embeddings in batches with rate limiting
to avoid hitting API limits.
"""

import unittest
from unittest.mock import MagicMock, patch, call
import time

from rag_system.ingestion.indexer import Indexer


class TestIndexerBatchedEmbeddings(unittest.TestCase):
    """Tests for Indexer._generate_embeddings with batching."""

    def setUp(self):
        """Set up mock vector client."""
        self.mock_vector_client = MagicMock()

    def test_processes_in_batches(self):
        """_generate_embeddings should process chunks in batches."""
        # Create mock that returns embeddings
        self.mock_vector_client.get_embeddings_batch.return_value = [[0.1, 0.2]]

        # Create chunks (more than one batch)
        chunks = [{'content': f'chunk {i}', 'heading_path': ''} for i in range(5)]
        chunk_ids = list(range(1, 6))

        # Mock config with small batch size
        with patch('rag_system.config.EMBEDDING_BATCH_SIZE', 2):
            with patch('rag_system.config.EMBEDDING_BATCH_DELAY', 0):
                from rag_system.ingestion.indexer import Indexer
                import tempfile
                import os
                from rag_system.database import init_db

                db_fd, db_path = tempfile.mkstemp()
                try:
                    init_db(db_path)
                    indexer = Indexer(db_path, vector_client=self.mock_vector_client)

                    # Mock update_chunk_embedding
                    with patch('rag_system.ingestion.indexer.update_chunk_embedding'):
                        from rag_system.database import get_connection
                        conn = get_connection(db_path)

                        # Call _generate_embeddings
                        indexer._generate_embeddings(conn, chunks, chunk_ids, 'Test Page')
                        conn.close()

                    # Should have been called 3 times (batches of 2, 2, 1)
                    self.assertEqual(
                        self.mock_vector_client.get_embeddings_batch.call_count, 3
                    )
                finally:
                    os.close(db_fd)
                    os.unlink(db_path)

    def test_handles_rate_limit_error(self):
        """_generate_embeddings should retry on rate limit (429)."""
        from rag_system.api_client import APIError

        # First call raises 429, second succeeds
        self.mock_vector_client.get_embeddings_batch.side_effect = [
            APIError("Rate limited", status_code=429),
            [[0.1, 0.2]]
        ]

        chunks = [{'content': 'chunk', 'heading_path': ''}]
        chunk_ids = [1]

        import tempfile
        import os
        from rag_system.database import init_db, get_connection

        db_fd, db_path = tempfile.mkstemp()
        try:
            init_db(db_path)
            indexer = Indexer(db_path, vector_client=self.mock_vector_client)

            with patch('rag_system.ingestion.indexer.update_chunk_embedding'):
                with patch('time.sleep'):  # Don't actually sleep in tests
                    conn = get_connection(db_path)
                    indexer._generate_embeddings(conn, chunks, chunk_ids, 'Test')
                    conn.close()

            # Should have retried
            self.assertEqual(
                self.mock_vector_client.get_embeddings_batch.call_count, 2
            )
        finally:
            os.close(db_fd)
            os.unlink(db_path)


class TestEmbeddingBatchConfig(unittest.TestCase):
    """Tests for embedding batch configuration."""

    def test_batch_size_config_exists(self):
        """EMBEDDING_BATCH_SIZE should be configurable."""
        from rag_system import config
        self.assertTrue(hasattr(config, 'EMBEDDING_BATCH_SIZE'))
        self.assertIsInstance(config.EMBEDDING_BATCH_SIZE, int)
        self.assertGreater(config.EMBEDDING_BATCH_SIZE, 0)

    def test_batch_delay_config_exists(self):
        """EMBEDDING_BATCH_DELAY should be configurable."""
        from rag_system import config
        self.assertTrue(hasattr(config, 'EMBEDDING_BATCH_DELAY'))
        self.assertIsInstance(config.EMBEDDING_BATCH_DELAY, (int, float))
        self.assertGreaterEqual(config.EMBEDDING_BATCH_DELAY, 0)


if __name__ == '__main__':
    unittest.main()
