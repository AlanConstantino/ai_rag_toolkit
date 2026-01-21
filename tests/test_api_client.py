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


if __name__ == '__main__':
    unittest.main()
