"""Tests for OpenAI API client.

Tests the OpenAI-compatible API clients for embeddings and chat completions.
"""

import json
import unittest
from unittest.mock import patch, MagicMock

from rag_system.api_client import OpenAIVectorClient, OpenAIChatClient


class TestOpenAIVectorClient(unittest.TestCase):
    """Tests for OpenAIVectorClient."""

    def test_init_with_api_key(self):
        """OpenAIVectorClient should initialize with API key."""
        client = OpenAIVectorClient(api_key='test-key')
        self.assertEqual(client.api_key, 'test-key')
        self.assertEqual(client.model, 'text-embedding-3-small')

    def test_init_with_custom_model(self):
        """OpenAIVectorClient should accept custom model."""
        client = OpenAIVectorClient(api_key='test-key', model='text-embedding-ada-002')
        self.assertEqual(client.model, 'text-embedding-ada-002')


    @patch('rag_system.api_client.urllib.request.urlopen')
    def test_get_embedding_success(self, mock_urlopen):
        """get_embedding should return embedding vector."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            'data': [{'embedding': [0.1, 0.2, 0.3]}],
            'model': 'text-embedding-3-small',
            'usage': {'total_tokens': 2}
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = OpenAIVectorClient(api_key='test-key')
        embedding = client.get_embedding('Hello')

        self.assertEqual(embedding, [0.1, 0.2, 0.3])

    @patch('rag_system.api_client.urllib.request.urlopen')
    def test_get_embeddings_batch_success(self, mock_urlopen):
        """get_embeddings_batch should return multiple embeddings."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            'data': [
                {'embedding': [0.1, 0.2, 0.3], 'index': 0},
                {'embedding': [0.4, 0.5, 0.6], 'index': 1}
            ],
            'model': 'text-embedding-3-small'
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = OpenAIVectorClient(api_key='test-key')
        embeddings = client.get_embeddings_batch(['Hello', 'World'])

        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3])
        self.assertEqual(embeddings[1], [0.4, 0.5, 0.6])


class TestOpenAIChatClient(unittest.TestCase):
    """Tests for OpenAIChatClient."""

    def test_init_with_api_key(self):
        """OpenAIChatClient should initialize with API key."""
        client = OpenAIChatClient(api_key='test-key')
        self.assertEqual(client.api_key, 'test-key')
        self.assertEqual(client.model, 'gpt-4o-mini')

    def test_init_with_custom_model(self):
        """OpenAIChatClient should accept custom model."""
        client = OpenAIChatClient(api_key='test-key', model='gpt-4o')
        self.assertEqual(client.model, 'gpt-4o')


    @patch('rag_system.api_client.urllib.request.urlopen')
    def test_complete_success(self, mock_urlopen):
        """complete should return chat response."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            'choices': [{'message': {'content': 'Hello there!'}}],
            'model': 'gpt-4o-mini'
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = OpenAIChatClient(api_key='test-key')
        response = client.complete('Hello')

        self.assertEqual(response, 'Hello there!')

    @patch('rag_system.api_client.urllib.request.urlopen')
    def test_complete_json_success(self, mock_urlopen):
        """complete_json should parse JSON response."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            'choices': [{'message': {'content': '{"key": "value"}'}}],
            'model': 'gpt-4o-mini'
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = OpenAIChatClient(api_key='test-key')
        response = client.complete_json('Return JSON')

        self.assertEqual(response, {'key': 'value'})


class TestOpenAIClientFactory(unittest.TestCase):
    """Tests for OpenAI client factory functions."""

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'})
    def test_create_openai_vector_client(self):
        """create_openai_vector_client should use env var."""
        from rag_system.api_client import create_openai_vector_client
        client = create_openai_vector_client()
        self.assertEqual(client.api_key, 'env-key')

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'})
    def test_create_openai_chat_client(self):
        """create_openai_chat_client should use env var."""
        from rag_system.api_client import create_openai_chat_client
        client = create_openai_chat_client()
        self.assertEqual(client.api_key, 'env-key')


if __name__ == '__main__':
    unittest.main()
