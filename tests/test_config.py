"""Tests for configuration validation.

Tests the validate_config and validate_or_raise functions.
"""

import os
import tempfile
import unittest
from unittest import mock


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation functions."""

    def test_default_config_is_valid(self):
        """Default configuration should pass validation without require_apis."""
        # Import fresh config module
        import importlib
        from rag_system import config
        importlib.reload(config)

        is_valid, errors = config.validate_config(require_apis=False)
        self.assertTrue(is_valid, f"Default config should be valid: {errors}")
        self.assertEqual(errors, [])

    def test_api_placeholder_check(self):
        """Placeholder API values should fail when require_apis=True."""
        import importlib
        from rag_system import config
        importlib.reload(config)

        is_valid, errors = config.validate_config(require_apis=True)
        self.assertFalse(is_valid)
        self.assertTrue(any('VECTOR_API_ENDPOINT' in e for e in errors))
        self.assertTrue(any('VECTOR_API_AUTH_VALUE' in e for e in errors))
        self.assertTrue(any('CHAT_API_ENDPOINT' in e for e in errors))
        self.assertTrue(any('CHAT_API_AUTH_VALUE' in e for e in errors))

    def test_configured_apis_pass_validation(self):
        """Properly configured APIs should pass validation."""
        env = {
            'RAG_VECTOR_API_ENDPOINT': 'https://api.openai.com/v1/embeddings',
            'RAG_VECTOR_API_AUTH_VALUE': 'Bearer sk-test123',
            'RAG_CHAT_API_ENDPOINT': 'https://api.openai.com/v1/chat/completions',
            'RAG_CHAT_API_AUTH_VALUE': 'Bearer sk-test456',
        }
        with mock.patch.dict(os.environ, env, clear=False):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config(require_apis=True)
            # May still have errors for other reasons but not for API placeholders
            api_errors = [e for e in errors if 'placeholder' in e.lower()]
            self.assertEqual(api_errors, [], f"Should not have API placeholder errors: {api_errors}")

    def test_negative_crawl_delay_fails(self):
        """Negative crawl delay should fail validation."""
        with mock.patch.dict(os.environ, {'RAG_CRAWL_DELAY_SECONDS': '-1.0'}):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config()
            self.assertFalse(is_valid)
            self.assertTrue(any('CRAWL_DELAY_SECONDS' in e for e in errors))

    def test_zero_max_pages_fails(self):
        """Zero max pages should fail validation."""
        with mock.patch.dict(os.environ, {'RAG_MAX_PAGES': '0'}):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config()
            self.assertFalse(is_valid)
            self.assertTrue(any('MAX_PAGES' in e for e in errors))

    def test_chunk_size_ordering(self):
        """Small chunk size must be <= large chunk size."""
        with mock.patch.dict(os.environ, {
            'RAG_SMALL_CHUNK_SIZE': '1000',
            'RAG_LARGE_CHUNK_SIZE': '500',
        }):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config()
            self.assertFalse(is_valid)
            self.assertTrue(any('SMALL_CHUNK_SIZE' in e and 'LARGE_CHUNK_SIZE' in e for e in errors))

    def test_chunk_overlap_less_than_chunk_size(self):
        """Chunk overlap must be less than small chunk size."""
        with mock.patch.dict(os.environ, {
            'RAG_CHUNK_OVERLAP': '600',
            'RAG_SMALL_CHUNK_SIZE': '500',
        }):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config()
            self.assertFalse(is_valid)
            self.assertTrue(any('CHUNK_OVERLAP' in e for e in errors))

    def test_weight_sum_validation(self):
        """Vector weight + BM25 weight must equal 1.0 when AI is enabled."""
        with mock.patch.dict(os.environ, {
            'RAG_AI_ENABLED': 'true',
            'RAG_VECTOR_WEIGHT': '0.8',
            'RAG_BM25_WEIGHT': '0.3',
        }):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config()
            self.assertFalse(is_valid)
            self.assertTrue(any('equal 1.0' in e for e in errors))

    def test_weight_sum_validation_skipped_when_ai_disabled(self):
        """Weight sum validation should be skipped when AI is disabled."""
        with mock.patch.dict(os.environ, {
            'RAG_AI_ENABLED': 'false',
            'RAG_VECTOR_WEIGHT': '0.8',
            'RAG_BM25_WEIGHT': '0.3',
        }):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config()
            # Should not have weight sum error when AI is disabled
            weight_errors = [e for e in errors if 'equal 1.0' in e]
            self.assertEqual(weight_errors, [])

    def test_ai_enabled_flag_default(self):
        """AI_ENABLED should default to False (BM25 mode)."""
        # Clear any RAG_AI_ENABLED env var
        env = {k: v for k, v in os.environ.items() if k != 'RAG_AI_ENABLED'}
        with mock.patch.dict(os.environ, env, clear=True):
            import importlib
            from rag_system import config
            importlib.reload(config)

            self.assertFalse(config.AI_ENABLED)

    def test_ai_enabled_flag_false(self):
        """AI_ENABLED should be False when set to false."""
        with mock.patch.dict(os.environ, {'RAG_AI_ENABLED': 'false'}):
            import importlib
            from rag_system import config
            importlib.reload(config)

            self.assertFalse(config.AI_ENABLED)

    def test_bm25_b_range(self):
        """BM25 B parameter must be between 0 and 1."""
        with mock.patch.dict(os.environ, {'RAG_BM25_B': '1.5'}):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config()
            self.assertFalse(is_valid)
            self.assertTrue(any('BM25_B' in e for e in errors))

    def test_confidence_score_range(self):
        """Confidence score must be between 1 and 5."""
        with mock.patch.dict(os.environ, {'RAG_MIN_CONFIDENCE_SCORE': '10'}):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config()
            self.assertFalse(is_valid)
            self.assertTrue(any('MIN_CONFIDENCE_SCORE' in e for e in errors))

    def test_top_k_final_less_than_retrieval(self):
        """TOP_K_FINAL must be <= TOP_K_RETRIEVAL."""
        with mock.patch.dict(os.environ, {
            'RAG_TOP_K_FINAL': '30',
            'RAG_TOP_K_RETRIEVAL': '20',
        }):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config()
            self.assertFalse(is_valid)
            self.assertTrue(any('TOP_K_FINAL' in e and 'TOP_K_RETRIEVAL' in e for e in errors))

    def test_ssl_cert_path_exists(self):
        """SSL cert path must exist if specified."""
        with mock.patch.dict(os.environ, {'RAG_SSL_CERT_PATH': '/nonexistent/path/cert.pem'}):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config()
            self.assertFalse(is_valid)
            self.assertTrue(any('SSL_CERT_PATH' in e for e in errors))

    def test_ssl_cert_path_valid(self):
        """Valid SSL cert path should pass validation."""
        with tempfile.NamedTemporaryFile(suffix='.pem', delete=False) as f:
            cert_path = f.name
            f.write(b'dummy cert content')

        try:
            with mock.patch.dict(os.environ, {'RAG_SSL_CERT_PATH': cert_path}):
                import importlib
                from rag_system import config
                importlib.reload(config)

                is_valid, errors = config.validate_config()
                ssl_errors = [e for e in errors if 'SSL_CERT_PATH' in e]
                self.assertEqual(ssl_errors, [])
        finally:
            os.unlink(cert_path)

    def test_validate_or_raise_success(self):
        """validate_or_raise should not raise for valid config."""
        import importlib
        from rag_system import config
        importlib.reload(config)

        # Should not raise
        config.validate_or_raise(require_apis=False)

    def test_validate_or_raise_failure(self):
        """validate_or_raise should raise ConfigurationError for invalid config."""
        with mock.patch.dict(os.environ, {'RAG_MAX_PAGES': '-1'}):
            import importlib
            from rag_system import config
            importlib.reload(config)

            with self.assertRaises(config.ConfigurationError) as ctx:
                config.validate_or_raise()

            self.assertIn('MAX_PAGES', str(ctx.exception))

    def test_token_chunking_validation_when_enabled(self):
        """Token chunking settings validated only when enabled."""
        # First, with token chunking disabled, invalid token settings should pass
        with mock.patch.dict(os.environ, {
            'RAG_USE_TOKEN_CHUNKING': 'false',
            'RAG_SMALL_CHUNK_TOKENS': '-1',
        }):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config()
            token_errors = [e for e in errors if 'CHUNK_TOKENS' in e]
            self.assertEqual(token_errors, [])

        # With token chunking enabled, invalid settings should fail
        with mock.patch.dict(os.environ, {
            'RAG_USE_TOKEN_CHUNKING': 'true',
            'RAG_SMALL_CHUNK_TOKENS': '-1',
        }):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config()
            self.assertFalse(is_valid)
            self.assertTrue(any('SMALL_CHUNK_TOKENS' in e for e in errors))

    def test_embedding_settings_validation(self):
        """Embedding settings should be validated."""
        with mock.patch.dict(os.environ, {'RAG_EMBEDDING_BATCH_SIZE': '0'}):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config()
            self.assertFalse(is_valid)
            self.assertTrue(any('EMBEDDING_BATCH_SIZE' in e for e in errors))

    def test_crawler_retry_settings_validation(self):
        """Crawler retry settings should be validated."""
        with mock.patch.dict(os.environ, {'RAG_CRAWLER_MAX_RETRIES': '-1'}):
            import importlib
            from rag_system import config
            importlib.reload(config)

            is_valid, errors = config.validate_config()
            self.assertFalse(is_valid)
            self.assertTrue(any('CRAWLER_MAX_RETRIES' in e for e in errors))


class TestConfigurationError(unittest.TestCase):
    """Test ConfigurationError exception."""

    def test_configuration_error_is_exception(self):
        """ConfigurationError should be an Exception."""
        from rag_system.config import ConfigurationError
        self.assertTrue(issubclass(ConfigurationError, Exception))

    def test_configuration_error_message(self):
        """ConfigurationError should have proper message."""
        from rag_system.config import ConfigurationError
        error = ConfigurationError("Test error message")
        self.assertEqual(str(error), "Test error message")


if __name__ == '__main__':
    unittest.main()
