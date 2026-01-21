"""Tests for the utils module."""

import unittest
import logging
import json
import tempfile
import os


class TestLogging(unittest.TestCase):
    """Test logging utilities."""

    def test_get_logger_returns_logger(self):
        """get_logger should return a configured logger."""
        from rag_system.utils import get_logger

        logger = get_logger('test_module')

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'test_module')

    def test_get_logger_with_level(self):
        """get_logger should support custom log levels."""
        from rag_system.utils import get_logger

        logger = get_logger('test_debug', level=logging.DEBUG)

        self.assertEqual(logger.level, logging.DEBUG)

    def test_get_logger_same_name_returns_same_logger(self):
        """get_logger should return the same logger for the same name."""
        from rag_system.utils import get_logger

        logger1 = get_logger('same_name')
        logger2 = get_logger('same_name')

        self.assertIs(logger1, logger2)


class TestJSONParsing(unittest.TestCase):
    """Test JSON parsing utilities."""

    def test_safe_json_loads_valid_json(self):
        """safe_json_loads should parse valid JSON."""
        from rag_system.utils import safe_json_loads

        result = safe_json_loads('{"key": "value", "num": 42}')

        self.assertEqual(result['key'], 'value')
        self.assertEqual(result['num'], 42)

    def test_safe_json_loads_invalid_json_returns_default(self):
        """safe_json_loads should return default for invalid JSON."""
        from rag_system.utils import safe_json_loads

        result = safe_json_loads('not valid json', default={})

        self.assertEqual(result, {})

    def test_safe_json_loads_with_custom_default(self):
        """safe_json_loads should use custom default."""
        from rag_system.utils import safe_json_loads

        result = safe_json_loads('invalid', default=['fallback'])

        self.assertEqual(result, ['fallback'])

    def test_safe_json_loads_extracts_json_from_text(self):
        """safe_json_loads should extract JSON from surrounding text."""
        from rag_system.utils import safe_json_loads

        text = 'Here is some text {"key": "value"} and more text'
        result = safe_json_loads(text)

        self.assertEqual(result['key'], 'value')

    def test_safe_json_loads_handles_json_in_markdown(self):
        """safe_json_loads should handle JSON in markdown code blocks."""
        from rag_system.utils import safe_json_loads

        text = '''Here is the JSON:
```json
{"entities": ["Redis", "MySQL"]}
```
'''
        result = safe_json_loads(text)

        self.assertEqual(result['entities'], ['Redis', 'MySQL'])

    def test_safe_json_loads_handles_array(self):
        """safe_json_loads should handle JSON arrays."""
        from rag_system.utils import safe_json_loads

        result = safe_json_loads('[1, 2, 3]')

        self.assertEqual(result, [1, 2, 3])


class TestHashingUtilities(unittest.TestCase):
    """Test hashing utilities."""

    def test_hash_content_returns_string(self):
        """hash_content should return a hex string."""
        from rag_system.utils import hash_content

        result = hash_content('test content')

        self.assertIsInstance(result, str)
        # SHA-256 produces 64 hex characters
        self.assertEqual(len(result), 64)

    def test_hash_content_is_deterministic(self):
        """hash_content should return the same hash for the same content."""
        from rag_system.utils import hash_content

        hash1 = hash_content('same content')
        hash2 = hash_content('same content')

        self.assertEqual(hash1, hash2)

    def test_hash_content_different_for_different_content(self):
        """hash_content should return different hashes for different content."""
        from rag_system.utils import hash_content

        hash1 = hash_content('content 1')
        hash2 = hash_content('content 2')

        self.assertNotEqual(hash1, hash2)


class TestTextUtilities(unittest.TestCase):
    """Test text processing utilities."""

    def test_tokenize_basic(self):
        """tokenize should split text into lowercase tokens."""
        from rag_system.utils import tokenize

        result = tokenize('Hello World')

        self.assertEqual(result, ['hello', 'world'])

    def test_tokenize_removes_punctuation(self):
        """tokenize should remove punctuation."""
        from rag_system.utils import tokenize

        result = tokenize("Hello, World! How's it going?")

        self.assertEqual(result, ['hello', 'world', 'hows', 'it', 'going'])

    def test_tokenize_handles_empty_string(self):
        """tokenize should handle empty strings."""
        from rag_system.utils import tokenize

        result = tokenize('')

        self.assertEqual(result, [])

    def test_tokenize_handles_numbers(self):
        """tokenize should keep numbers."""
        from rag_system.utils import tokenize

        result = tokenize('Port 8080 is used')

        self.assertEqual(result, ['port', '8080', 'is', 'used'])

    def test_clean_text_removes_extra_whitespace(self):
        """clean_text should normalize whitespace."""
        from rag_system.utils import clean_text

        result = clean_text('Hello   world\n\ntest')

        self.assertEqual(result, 'Hello world test')

    def test_clean_text_strips_text(self):
        """clean_text should strip leading and trailing whitespace."""
        from rag_system.utils import clean_text

        result = clean_text('  Hello world  ')

        self.assertEqual(result, 'Hello world')

    def test_truncate_text_short_text(self):
        """truncate_text should not modify short text."""
        from rag_system.utils import truncate_text

        result = truncate_text('short', max_length=100)

        self.assertEqual(result, 'short')

    def test_truncate_text_long_text(self):
        """truncate_text should truncate long text with ellipsis."""
        from rag_system.utils import truncate_text

        result = truncate_text('This is a longer text', max_length=10)

        self.assertEqual(len(result), 10)
        self.assertTrue(result.endswith('...'))


class TestQueryHashUtility(unittest.TestCase):
    """Test query hash utility."""

    def test_get_query_hash_returns_string(self):
        """get_query_hash should return a hash string."""
        from rag_system.utils import get_query_hash

        result = get_query_hash('test query')

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_get_query_hash_normalizes_query(self):
        """get_query_hash should normalize queries before hashing."""
        from rag_system.utils import get_query_hash

        hash1 = get_query_hash('  Test Query  ')
        hash2 = get_query_hash('test query')

        self.assertEqual(hash1, hash2)


class TestTimingUtility(unittest.TestCase):
    """Test timing utilities."""

    def test_timer_context_manager(self):
        """Timer context manager should measure elapsed time."""
        from rag_system.utils import Timer
        import time

        with Timer() as t:
            time.sleep(0.01)  # Sleep 10ms

        self.assertGreater(t.elapsed, 0.01)
        self.assertLess(t.elapsed, 0.1)  # Should be less than 100ms


class TestRetryUtility(unittest.TestCase):
    """Test retry utilities."""

    def test_retry_succeeds_first_try(self):
        """retry should return result on first successful try."""
        from rag_system.utils import retry

        call_count = [0]

        @retry(max_attempts=3)
        def successful_func():
            call_count[0] += 1
            return 'success'

        result = successful_func()

        self.assertEqual(result, 'success')
        self.assertEqual(call_count[0], 1)

    def test_retry_retries_on_exception(self):
        """retry should retry on exception."""
        from rag_system.utils import retry

        call_count = [0]

        @retry(max_attempts=3, delay=0.01)
        def failing_then_success():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError('Not yet')
            return 'success'

        result = failing_then_success()

        self.assertEqual(result, 'success')
        self.assertEqual(call_count[0], 3)

    def test_retry_raises_after_max_attempts(self):
        """retry should raise after max attempts."""
        from rag_system.utils import retry

        @retry(max_attempts=2, delay=0.01)
        def always_fails():
            raise ValueError('Always fails')

        with self.assertRaises(ValueError):
            always_fails()


if __name__ == '__main__':
    unittest.main()
