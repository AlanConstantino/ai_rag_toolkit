"""Tests for OpenAI API client.

Tests the OpenAI-compatible API clients for embeddings and chat completions.
Uses real API calls per project policy (no mocks).
"""

import os
import unittest

from rag_system.api_client import (
    OpenAIVectorClient, OpenAIChatClient,
    create_openai_vector_client, create_openai_chat_client
)


def get_api_key():
    """Get OpenAI API key from environment."""
    return os.environ.get('OPENAI_API_KEY')


class TestOpenAIVectorClientInit(unittest.TestCase):
    """Tests for OpenAIVectorClient initialization."""

    def test_init_with_api_key(self):
        """OpenAIVectorClient should initialize with API key."""
        client = OpenAIVectorClient(api_key='test-key')
        self.assertEqual(client.api_key, 'test-key')
        self.assertEqual(client.model, 'text-embedding-3-small')

    def test_init_with_custom_model(self):
        """OpenAIVectorClient should accept custom model."""
        client = OpenAIVectorClient(api_key='test-key', model='text-embedding-ada-002')
        self.assertEqual(client.model, 'text-embedding-ada-002')


class TestOpenAIChatClientInit(unittest.TestCase):
    """Tests for OpenAIChatClient initialization."""

    def test_init_with_api_key(self):
        """OpenAIChatClient should initialize with API key."""
        client = OpenAIChatClient(api_key='test-key')
        self.assertEqual(client.api_key, 'test-key')
        self.assertEqual(client.model, 'gpt-4o-mini')

    def test_init_with_custom_model(self):
        """OpenAIChatClient should accept custom model."""
        client = OpenAIChatClient(api_key='test-key', model='gpt-4o')
        self.assertEqual(client.model, 'gpt-4o')


@unittest.skipUnless(get_api_key(), 'OPENAI_API_KEY not set')
class TestOpenAIVectorClientAPI(unittest.TestCase):
    """Tests for OpenAIVectorClient API calls (requires API key)."""

    def setUp(self):
        """Set up client with real API key."""
        self.client = OpenAIVectorClient(api_key=get_api_key())

    def test_get_embedding_returns_vector(self):
        """get_embedding should return a non-empty embedding vector."""
        embedding = self.client.get_embedding('Hello world')

        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        self.assertIsInstance(embedding[0], float)

    def test_get_embeddings_batch_returns_multiple(self):
        """get_embeddings_batch should return embeddings for all inputs."""
        texts = ['Hello', 'World', 'Test']
        embeddings = self.client.get_embeddings_batch(texts)

        self.assertEqual(len(embeddings), 3)
        for embedding in embeddings:
            self.assertIsInstance(embedding, list)
            self.assertGreater(len(embedding), 0)

    def test_embedding_dimensions_consistent(self):
        """Embeddings should have consistent dimensions."""
        e1 = self.client.get_embedding('First text')
        e2 = self.client.get_embedding('Second text')

        self.assertEqual(len(e1), len(e2))


@unittest.skipUnless(get_api_key(), 'OPENAI_API_KEY not set')
class TestOpenAIChatClientAPI(unittest.TestCase):
    """Tests for OpenAIChatClient API calls (requires API key)."""

    def setUp(self):
        """Set up client with real API key."""
        self.client = OpenAIChatClient(api_key=get_api_key())

    def test_complete_returns_string(self):
        """complete should return a non-empty string response."""
        response = self.client.complete('Say hello in one word.')

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_complete_json_returns_dict(self):
        """complete_json should parse and return JSON."""
        response = self.client.complete_json(
            'Return a JSON object with a single key "greeting" and value "hello". '
            'Return only the JSON, no explanation.'
        )

        self.assertIsInstance(response, dict)
        self.assertIn('greeting', response)


@unittest.skipUnless(get_api_key(), 'OPENAI_API_KEY not set')
class TestOpenAIClientFactory(unittest.TestCase):
    """Tests for OpenAI client factory functions (requires API key)."""

    def test_create_openai_vector_client_uses_env(self):
        """create_openai_vector_client should use OPENAI_API_KEY env var."""
        client = create_openai_vector_client()
        self.assertEqual(client.api_key, get_api_key())

    def test_create_openai_chat_client_uses_env(self):
        """create_openai_chat_client should use OPENAI_API_KEY env var."""
        client = create_openai_chat_client()
        self.assertEqual(client.api_key, get_api_key())


if __name__ == '__main__':
    unittest.main()
