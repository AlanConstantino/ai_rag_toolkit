"""Tests for the API client module."""

import unittest
from unittest.mock import patch, MagicMock
import json


class TestVectorAPIClient(unittest.TestCase):
    """Test vector API client."""

    def test_get_embedding_returns_list(self):
        """get_embedding should return a list of floats."""
        from rag_system.api_client import VectorAPIClient

        # Mock the HTTP request
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            'embedding': [0.1, 0.2, 0.3, 0.4]
        }).encode('utf-8')
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            client = VectorAPIClient(
                endpoint='https://api.example.com/embed',
                auth_header='Authorization',
                auth_value='Bearer token'
            )
            embedding = client.get_embedding('test text')

        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 4)
        self.assertAlmostEqual(embedding[0], 0.1)

    def test_get_embeddings_batch_returns_list_of_lists(self):
        """get_embeddings_batch should return a list of embeddings."""
        from rag_system.api_client import VectorAPIClient

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            'embeddings': [[0.1, 0.2], [0.3, 0.4]]
        }).encode('utf-8')
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            client = VectorAPIClient(
                endpoint='https://api.example.com/embed',
                auth_header='Authorization',
                auth_value='Bearer token'
            )
            embeddings = client.get_embeddings_batch(['text1', 'text2'])

        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 2)

    def test_handles_api_error(self):
        """get_embedding should raise APIError on failure."""
        from rag_system.api_client import VectorAPIClient, APIError
        import urllib.error

        with patch('urllib.request.urlopen',
                   side_effect=urllib.error.HTTPError(
                       'url', 500, 'Server Error', {}, None)):
            client = VectorAPIClient(
                endpoint='https://api.example.com/embed',
                auth_header='Authorization',
                auth_value='Bearer token'
            )
            with self.assertRaises(APIError):
                client.get_embedding('test text')

    def test_creates_correct_request(self):
        """get_embedding should create request with correct headers."""
        from rag_system.api_client import VectorAPIClient

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            'embedding': [0.1]
        }).encode('utf-8')
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response) as mock_urlopen:
            client = VectorAPIClient(
                endpoint='https://api.example.com/embed',
                auth_header='X-API-Key',
                auth_value='my-key'
            )
            client.get_embedding('test')

            # Verify request was made with correct headers
            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            self.assertEqual(request.get_header('X-api-key'), 'my-key')
            self.assertEqual(request.get_header('Content-type'), 'application/json')


class TestChatAPIClient(unittest.TestCase):
    """Test chat API client."""

    def test_complete_returns_string(self):
        """complete should return a string response."""
        from rag_system.api_client import ChatAPIClient

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            'response': 'This is the answer.'
        }).encode('utf-8')
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            client = ChatAPIClient(
                endpoint='https://api.example.com/chat',
                auth_header='Authorization',
                auth_value='Bearer token'
            )
            response = client.complete('What is 2+2?')

        self.assertIsInstance(response, str)
        self.assertEqual(response, 'This is the answer.')

    def test_complete_with_system_prompt(self):
        """complete should support system prompts."""
        from rag_system.api_client import ChatAPIClient

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            'response': 'Answer with system context.'
        }).encode('utf-8')
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response) as mock_urlopen:
            client = ChatAPIClient(
                endpoint='https://api.example.com/chat',
                auth_header='Authorization',
                auth_value='Bearer token'
            )
            response = client.complete(
                'Question',
                system_prompt='You are a helpful assistant.'
            )

            # Verify system prompt was included in request
            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            body = json.loads(request.data.decode('utf-8'))
            self.assertIn('system_prompt', body)
            self.assertEqual(body['system_prompt'], 'You are a helpful assistant.')

    def test_handles_api_error(self):
        """complete should raise APIError on failure."""
        from rag_system.api_client import ChatAPIClient, APIError
        import urllib.error

        with patch('urllib.request.urlopen',
                   side_effect=urllib.error.HTTPError(
                       'url', 500, 'Server Error', {}, None)):
            client = ChatAPIClient(
                endpoint='https://api.example.com/chat',
                auth_header='Authorization',
                auth_value='Bearer token'
            )
            with self.assertRaises(APIError):
                client.complete('test prompt')

    def test_complete_json_returns_parsed_json(self):
        """complete_json should return parsed JSON response."""
        from rag_system.api_client import ChatAPIClient

        # The API returns JSON in a response field
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            'response': '{"entities": ["Redis", "MySQL"], "count": 2}'
        }).encode('utf-8')
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            client = ChatAPIClient(
                endpoint='https://api.example.com/chat',
                auth_header='Authorization',
                auth_value='Bearer token'
            )
            result = client.complete_json('Extract entities as JSON')

        self.assertIsInstance(result, dict)
        self.assertEqual(result['entities'], ['Redis', 'MySQL'])
        self.assertEqual(result['count'], 2)

    def test_complete_json_handles_invalid_json(self):
        """complete_json should raise ValueError for invalid JSON."""
        from rag_system.api_client import ChatAPIClient

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            'response': 'This is not valid JSON'
        }).encode('utf-8')
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            client = ChatAPIClient(
                endpoint='https://api.example.com/chat',
                auth_header='Authorization',
                auth_value='Bearer token'
            )
            with self.assertRaises(ValueError):
                client.complete_json('prompt')


class TestAPIClientFactory(unittest.TestCase):
    """Test API client factory functions."""

    def test_create_vector_client_from_config(self):
        """create_vector_client should use config settings."""
        from rag_system.api_client import create_vector_client, VectorAPIClient

        with patch('rag_system.api_client.config') as mock_config:
            mock_config.VECTOR_API_ENDPOINT = 'https://vector.api.com'
            mock_config.VECTOR_API_AUTH_HEADER = 'X-API-Key'
            mock_config.VECTOR_API_AUTH_VALUE = 'secret'
            mock_config.SSL_VERIFY = True
            mock_config.SSL_CERT_PATH = None

            client = create_vector_client()

        self.assertIsInstance(client, VectorAPIClient)

    def test_create_chat_client_from_config(self):
        """create_chat_client should use config settings."""
        from rag_system.api_client import create_chat_client, ChatAPIClient

        with patch('rag_system.api_client.config') as mock_config:
            mock_config.CHAT_API_ENDPOINT = 'https://chat.api.com'
            mock_config.CHAT_API_AUTH_HEADER = 'Authorization'
            mock_config.CHAT_API_AUTH_VALUE = 'Bearer token'
            mock_config.SSL_VERIFY = True
            mock_config.SSL_CERT_PATH = None

            client = create_chat_client()

        self.assertIsInstance(client, ChatAPIClient)


class TestSSLConfiguration(unittest.TestCase):
    """Test SSL configuration for API clients."""

    def test_ssl_context_created_when_configured(self):
        """API client should use SSL context when SSL_CERT_PATH is set."""
        from rag_system.api_client import VectorAPIClient
        import ssl

        with patch('ssl.create_default_context') as mock_ssl:
            mock_ctx = MagicMock(spec=ssl.SSLContext)
            mock_ssl.return_value = mock_ctx

            client = VectorAPIClient(
                endpoint='https://api.example.com/embed',
                auth_header='Authorization',
                auth_value='Bearer token',
                ssl_verify=True,
                ssl_cert_path='/path/to/cert.pem'
            )

            mock_ssl.assert_called_once()
            mock_ctx.load_verify_locations.assert_called_once_with('/path/to/cert.pem')

    def test_ssl_verification_can_be_disabled(self):
        """API client should allow disabling SSL verification."""
        from rag_system.api_client import VectorAPIClient

        # Should not raise even with ssl_verify=False
        client = VectorAPIClient(
            endpoint='https://api.example.com/embed',
            auth_header='Authorization',
            auth_value='Bearer token',
            ssl_verify=False
        )

        self.assertIsNotNone(client)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality."""

    def test_circuit_breaker_starts_closed(self):
        """Circuit breaker should start in closed state."""
        from rag_system.api_client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, reset_timeout=1.0)
        self.assertEqual(cb.state, CircuitBreaker.CLOSED)
        self.assertTrue(cb.can_execute())

    def test_circuit_breaker_opens_after_threshold(self):
        """Circuit breaker should open after failure threshold."""
        from rag_system.api_client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60.0)

        # Record failures up to threshold
        cb.record_failure()
        self.assertTrue(cb.can_execute())
        cb.record_failure()
        self.assertTrue(cb.can_execute())
        cb.record_failure()  # Threshold reached

        self.assertEqual(cb.state, CircuitBreaker.OPEN)
        self.assertFalse(cb.can_execute())

    def test_circuit_breaker_closes_on_success(self):
        """Circuit breaker should close on success after being open."""
        from rag_system.api_client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.01)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        self.assertFalse(cb.can_execute())

        # Wait for reset timeout
        import time
        time.sleep(0.02)

        # Should be half-open now
        self.assertEqual(cb.state, CircuitBreaker.HALF_OPEN)
        self.assertTrue(cb.can_execute())

        # Success should close it
        cb.record_success()
        self.assertEqual(cb.state, CircuitBreaker.CLOSED)

    def test_circuit_breaker_reset(self):
        """reset() should return circuit breaker to closed state."""
        from rag_system.api_client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        self.assertFalse(cb.can_execute())

        cb.reset()
        self.assertEqual(cb.state, CircuitBreaker.CLOSED)
        self.assertTrue(cb.can_execute())

    def test_success_resets_failure_count(self):
        """Success should reset failure count."""
        from rag_system.api_client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # Reset failure count
        cb.record_failure()
        cb.record_failure()  # Still under threshold

        self.assertTrue(cb.can_execute())


class TestCircuitBreakerIntegration(unittest.TestCase):
    """Test circuit breaker integration with API clients."""

    def test_vector_client_uses_circuit_breaker(self):
        """VectorAPIClient should respect circuit breaker."""
        from rag_system.api_client import (
            VectorAPIClient, CircuitBreaker, CircuitBreakerError
        )
        import urllib.error

        # Create a custom circuit breaker with low threshold
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60.0)
        client = VectorAPIClient(
            endpoint='https://api.example.com/embed',
            auth_header='Authorization',
            auth_value='Bearer token',
            circuit_breaker=cb
        )

        # Force circuit to open
        with patch('urllib.request.urlopen',
                   side_effect=urllib.error.HTTPError('url', 500, 'Error', {}, None)):
            try:
                client.get_embedding('test')
            except Exception:
                pass
            try:
                client.get_embedding('test')
            except Exception:
                pass

        # Now circuit should be open
        self.assertFalse(cb.can_execute())

        # Attempting another request should raise CircuitBreakerError
        with self.assertRaises(CircuitBreakerError):
            client.get_embedding('test')

    def test_chat_client_uses_circuit_breaker(self):
        """ChatAPIClient should respect circuit breaker."""
        from rag_system.api_client import (
            ChatAPIClient, CircuitBreaker, CircuitBreakerError
        )
        import urllib.error

        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60.0)
        client = ChatAPIClient(
            endpoint='https://api.example.com/chat',
            auth_header='Authorization',
            auth_value='Bearer token',
            circuit_breaker=cb
        )

        # Force circuit to open
        with patch('urllib.request.urlopen',
                   side_effect=urllib.error.HTTPError('url', 500, 'Error', {}, None)):
            try:
                client.complete('test')
            except Exception:
                pass
            try:
                client.complete('test')
            except Exception:
                pass

        # Circuit should be open
        self.assertFalse(cb.can_execute())

        with self.assertRaises(CircuitBreakerError):
            client.complete('test')

    def test_client_is_available_method(self):
        """Clients should have is_available() method."""
        from rag_system.api_client import VectorAPIClient, CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1)
        client = VectorAPIClient(
            endpoint='https://api.example.com/embed',
            auth_header='Authorization',
            auth_value='Bearer token',
            circuit_breaker=cb
        )

        self.assertTrue(client.is_available())
        cb.record_failure()
        self.assertFalse(client.is_available())

    def test_client_reset_circuit_breaker_method(self):
        """Clients should have reset_circuit_breaker() method."""
        from rag_system.api_client import ChatAPIClient, CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1)
        client = ChatAPIClient(
            endpoint='https://api.example.com/chat',
            auth_header='Authorization',
            auth_value='Bearer token',
            circuit_breaker=cb
        )

        cb.record_failure()
        self.assertFalse(client.is_available())

        client.reset_circuit_breaker()
        self.assertTrue(client.is_available())


class TestCustomExceptions(unittest.TestCase):
    """Test custom exception classes."""

    def test_api_error_has_attributes(self):
        """APIError should store status_code and response."""
        from rag_system.api_client import APIError

        error = APIError("Test error", status_code=500, response='{"error": "Server Error"}')
        self.assertEqual(str(error), "Test error")
        self.assertEqual(error.status_code, 500)
        self.assertEqual(error.response, '{"error": "Server Error"}')

    def test_circuit_breaker_error_is_api_error(self):
        """CircuitBreakerError should be subclass of APIError."""
        from rag_system.api_client import APIError, CircuitBreakerError

        self.assertTrue(issubclass(CircuitBreakerError, APIError))

    def test_timeout_error_is_api_error(self):
        """TimeoutError should be subclass of APIError."""
        from rag_system.api_client import APIError, TimeoutError

        self.assertTrue(issubclass(TimeoutError, APIError))


class TestRetryBehavior(unittest.TestCase):
    """Test HTTP request retry behavior."""

    def test_retries_on_server_error(self):
        """make_http_request should retry on 5xx errors."""
        from rag_system.api_client import make_http_request, APIError
        import urllib.error

        call_count = [0]

        def mock_urlopen(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise urllib.error.HTTPError('url', 503, 'Service Unavailable', {}, None)
            # Success on 3rd try
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"result": "success"}'
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            return mock_response

        with patch('urllib.request.urlopen', side_effect=mock_urlopen):
            with patch('time.sleep'):  # Skip actual delay
                result = make_http_request(
                    'https://api.example.com',
                    {'data': 'test'},
                    {'Content-Type': 'application/json'},
                    max_retries=3
                )

        self.assertEqual(result['result'], 'success')
        self.assertEqual(call_count[0], 3)

    def test_raises_after_max_retries(self):
        """make_http_request should raise after exhausting retries."""
        from rag_system.api_client import make_http_request, APIError
        import urllib.error

        with patch('urllib.request.urlopen',
                   side_effect=urllib.error.HTTPError('url', 503, 'Error', {}, None)):
            with patch('time.sleep'):
                with self.assertRaises(APIError) as ctx:
                    make_http_request(
                        'https://api.example.com',
                        {'data': 'test'},
                        {'Content-Type': 'application/json'},
                        max_retries=2
                    )

                self.assertEqual(ctx.exception.status_code, 503)


if __name__ == '__main__':
    unittest.main()
