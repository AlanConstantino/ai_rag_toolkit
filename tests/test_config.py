"""Tests for the config module."""

import unittest
import os
import tempfile


class TestConfig(unittest.TestCase):
    """Test configuration settings."""

    def test_config_has_vector_api_endpoint(self):
        """Config should have VECTOR_API_ENDPOINT setting."""
        from rag_system import config
        self.assertTrue(hasattr(config, 'VECTOR_API_ENDPOINT'))
        self.assertIsInstance(config.VECTOR_API_ENDPOINT, str)

    def test_config_has_vector_api_auth(self):
        """Config should have vector API authentication settings."""
        from rag_system import config
        self.assertTrue(hasattr(config, 'VECTOR_API_AUTH_HEADER'))
        self.assertTrue(hasattr(config, 'VECTOR_API_AUTH_VALUE'))

    def test_config_has_chat_api_endpoint(self):
        """Config should have CHAT_API_ENDPOINT setting."""
        from rag_system import config
        self.assertTrue(hasattr(config, 'CHAT_API_ENDPOINT'))
        self.assertIsInstance(config.CHAT_API_ENDPOINT, str)

    def test_config_has_chat_api_auth(self):
        """Config should have chat API authentication settings."""
        from rag_system import config
        self.assertTrue(hasattr(config, 'CHAT_API_AUTH_HEADER'))
        self.assertTrue(hasattr(config, 'CHAT_API_AUTH_VALUE'))

    def test_config_has_ssl_settings(self):
        """Config should have SSL configuration settings."""
        from rag_system import config
        self.assertTrue(hasattr(config, 'SSL_CERT_PATH'))
        self.assertTrue(hasattr(config, 'SSL_VERIFY'))

    def test_config_has_database_path(self):
        """Config should have DATABASE_PATH setting."""
        from rag_system import config
        self.assertTrue(hasattr(config, 'DATABASE_PATH'))
        self.assertIsInstance(config.DATABASE_PATH, str)

    def test_config_has_crawler_settings(self):
        """Config should have crawler settings."""
        from rag_system import config
        self.assertTrue(hasattr(config, 'CRAWL_DELAY_SECONDS'))
        self.assertTrue(hasattr(config, 'MAX_PAGES'))
        self.assertTrue(hasattr(config, 'ALLOWED_DOMAINS'))
        self.assertTrue(hasattr(config, 'EXCLUDED_PATHS'))
        self.assertIsInstance(config.CRAWL_DELAY_SECONDS, (int, float))
        self.assertIsInstance(config.MAX_PAGES, int)
        self.assertIsInstance(config.ALLOWED_DOMAINS, list)
        self.assertIsInstance(config.EXCLUDED_PATHS, list)

    def test_config_has_chunking_settings(self):
        """Config should have chunking settings."""
        from rag_system import config
        self.assertTrue(hasattr(config, 'SMALL_CHUNK_SIZE'))
        self.assertTrue(hasattr(config, 'LARGE_CHUNK_SIZE'))
        self.assertTrue(hasattr(config, 'CHUNK_OVERLAP'))
        self.assertIsInstance(config.SMALL_CHUNK_SIZE, int)
        self.assertIsInstance(config.LARGE_CHUNK_SIZE, int)
        self.assertIsInstance(config.CHUNK_OVERLAP, int)
        self.assertGreater(config.LARGE_CHUNK_SIZE, config.SMALL_CHUNK_SIZE)

    def test_config_has_search_settings(self):
        """Config should have search settings."""
        from rag_system import config
        self.assertTrue(hasattr(config, 'BM25_K1'))
        self.assertTrue(hasattr(config, 'BM25_B'))
        self.assertTrue(hasattr(config, 'VECTOR_WEIGHT'))
        self.assertTrue(hasattr(config, 'BM25_WEIGHT'))
        self.assertTrue(hasattr(config, 'TOP_K_RETRIEVAL'))
        self.assertTrue(hasattr(config, 'TOP_K_FINAL'))
        self.assertTrue(hasattr(config, 'MAX_CHUNKS_PER_PAGE'))

    def test_config_has_confidence_threshold(self):
        """Config should have confidence threshold setting."""
        from rag_system import config
        self.assertTrue(hasattr(config, 'MIN_CONFIDENCE_SCORE'))
        self.assertIsInstance(config.MIN_CONFIDENCE_SCORE, (int, float))
        self.assertGreaterEqual(config.MIN_CONFIDENCE_SCORE, 1)
        self.assertLessEqual(config.MIN_CONFIDENCE_SCORE, 5)

    def test_search_weights_sum_to_one(self):
        """Vector and BM25 weights should sum to 1.0."""
        from rag_system import config
        total = config.VECTOR_WEIGHT + config.BM25_WEIGHT
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_bm25_parameters_in_valid_range(self):
        """BM25 parameters should be in typical valid ranges."""
        from rag_system import config
        # K1 typically between 1.2 and 2.0
        self.assertGreaterEqual(config.BM25_K1, 0)
        self.assertLessEqual(config.BM25_K1, 3.0)
        # B typically 0.75
        self.assertGreaterEqual(config.BM25_B, 0)
        self.assertLessEqual(config.BM25_B, 1.0)


class TestConfigEnvironmentOverrides(unittest.TestCase):
    """Test that config can be overridden by environment variables."""

    def setUp(self):
        """Store original environment."""
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
        # Force reload of config module
        import importlib
        from rag_system import config
        importlib.reload(config)

    def test_database_path_can_be_overridden(self):
        """DATABASE_PATH should be overridable via environment variable."""
        os.environ['RAG_DATABASE_PATH'] = '/tmp/test_rag.db'
        import importlib
        from rag_system import config
        importlib.reload(config)
        self.assertEqual(config.DATABASE_PATH, '/tmp/test_rag.db')


if __name__ == '__main__':
    unittest.main()
