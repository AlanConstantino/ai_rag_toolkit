"""API client module for the RAG system.

Provides wrappers for vector embedding and chat completion APIs.
Includes OpenAI-compatible clients for use with OpenAI's API.
Uses only Python standard library (urllib).
"""

import json
import os
import re
import ssl
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any, Union

from rag_system import config


class APIError(Exception):
    """Exception raised when an API call fails."""

    def __init__(self, message: str, status_code: Optional[int] = None,
                 response: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


# =============================================================================
# Shared HTTP Utilities
# =============================================================================

def create_ssl_context(ssl_verify: bool, ssl_cert_path: Optional[str]) -> Optional[ssl.SSLContext]:
    """Create SSL context for HTTPS requests.

    Args:
        ssl_verify: Whether to verify SSL certificates.
        ssl_cert_path: Path to SSL certificate file.

    Returns:
        SSL context or None.
    """
    if not ssl_verify:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    if ssl_cert_path:
        ctx = ssl.create_default_context()
        ctx.load_verify_locations(ssl_cert_path)
        return ctx

    return None


def make_http_request(endpoint: str, data: Dict[str, Any], headers: Dict[str, str],
                      ssl_context: Optional[ssl.SSLContext] = None) -> Dict[str, Any]:
    """Make an HTTP POST request and return parsed JSON.

    Args:
        endpoint: API endpoint URL.
        data: Request body data.
        headers: HTTP headers.
        ssl_context: Optional SSL context.

    Returns:
        Parsed JSON response.

    Raises:
        APIError: If the request fails.
    """
    body = json.dumps(data).encode('utf-8')
    request = urllib.request.Request(endpoint, data=body, headers=headers, method='POST')

    try:
        with urllib.request.urlopen(request, context=ssl_context) as response:
            response_data = response.read().decode('utf-8')
            return json.loads(response_data)
    except urllib.error.HTTPError as e:
        raise APIError(
            f"API request failed: {e.reason}",
            status_code=e.code,
            response=e.read().decode('utf-8') if e.fp else None
        )
    except urllib.error.URLError as e:
        raise APIError(f"API request failed: {e.reason}")


class VectorAPIClient:
    """Client for vector embedding API."""

    def __init__(self, endpoint: str, auth_header: str, auth_value: str,
                 ssl_verify: bool = True, ssl_cert_path: Optional[str] = None):
        """Initialize the vector API client.

        Args:
            endpoint: API endpoint URL.
            auth_header: Name of the authentication header.
            auth_value: Value for the authentication header.
            ssl_verify: Whether to verify SSL certificates.
            ssl_cert_path: Path to SSL certificate file.
        """
        self.endpoint = endpoint
        self.auth_header = auth_header
        self.auth_value = auth_value
        self.ssl_context = create_ssl_context(ssl_verify, ssl_cert_path)

    def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP request to the API."""
        headers = {
            'Content-Type': 'application/json',
            self.auth_header: self.auth_value
        }
        return make_http_request(self.endpoint, data, headers, self.ssl_context)

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            APIError: If the request fails.
        """
        response = self._make_request({'text': text})
        return response.get('embedding', [])

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            APIError: If the request fails.
        """
        response = self._make_request({'texts': texts})
        return response.get('embeddings', [])


class ChatAPIClient:
    """Client for chat completion API."""

    def __init__(self, endpoint: str, auth_header: str, auth_value: str,
                 ssl_verify: bool = True, ssl_cert_path: Optional[str] = None):
        """Initialize the chat API client.

        Args:
            endpoint: API endpoint URL.
            auth_header: Name of the authentication header.
            auth_value: Value for the authentication header.
            ssl_verify: Whether to verify SSL certificates.
            ssl_cert_path: Path to SSL certificate file.
        """
        self.endpoint = endpoint
        self.auth_header = auth_header
        self.auth_value = auth_value
        self.ssl_context = create_ssl_context(ssl_verify, ssl_cert_path)

    def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP request to the API."""
        headers = {
            'Content-Type': 'application/json',
            self.auth_header: self.auth_value
        }
        return make_http_request(self.endpoint, data, headers, self.ssl_context)

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Get a chat completion.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt for context.

        Returns:
            Completion response as a string.

        Raises:
            APIError: If the request fails.
        """
        data = {'prompt': prompt}
        if system_prompt:
            data['system_prompt'] = system_prompt

        response = self._make_request(data)
        return response.get('response', '')

    def complete_json(self, prompt: str,
                      system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Get a chat completion and parse it as JSON.

        Args:
            prompt: User prompt (should request JSON output).
            system_prompt: Optional system prompt for context.

        Returns:
            Parsed JSON response.

        Raises:
            APIError: If the request fails.
            ValueError: If the response is not valid JSON.
        """
        response = self.complete(prompt, system_prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response as JSON: {e}")


# =============================================================================
# OpenAI API Clients
# =============================================================================

class OpenAIVectorClient:
    """Client for OpenAI embeddings API."""

    ENDPOINT = 'https://api.openai.com/v1/embeddings'

    def __init__(self, api_key: str, model: str = 'text-embedding-3-small'):
        """Initialize the OpenAI vector client.

        Args:
            api_key: OpenAI API key.
            model: Embedding model to use.
        """
        self.api_key = api_key
        self.model = model

    def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP request to OpenAI API."""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        return make_http_request(self.ENDPOINT, data, headers)

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            APIError: If the request fails.
        """
        data = {'input': text, 'model': self.model}
        response = self._make_request(data)
        return response['data'][0]['embedding']

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in the same order as input.

        Raises:
            APIError: If the request fails.
        """
        data = {'input': texts, 'model': self.model}
        response = self._make_request(data)
        sorted_data = sorted(response['data'], key=lambda x: x['index'])
        return [item['embedding'] for item in sorted_data]


class OpenAIChatClient:
    """Client for OpenAI chat completions API."""

    ENDPOINT = 'https://api.openai.com/v1/chat/completions'

    def __init__(self, api_key: str, model: str = 'gpt-4o-mini'):
        """Initialize the OpenAI chat client.

        Args:
            api_key: OpenAI API key.
            model: Chat model to use.
        """
        self.api_key = api_key
        self.model = model

    def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP request to OpenAI API."""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        return make_http_request(self.ENDPOINT, data, headers)

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Get a chat completion.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt for context.

        Returns:
            Completion response as a string.

        Raises:
            APIError: If the request fails.
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        data = {'model': self.model, 'messages': messages}
        response = self._make_request(data)
        return response['choices'][0]['message']['content']

    def complete_json(self, prompt: str,
                      system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Get a chat completion and parse it as JSON.

        Args:
            prompt: User prompt (should request JSON output).
            system_prompt: Optional system prompt for context.

        Returns:
            Parsed JSON response.

        Raises:
            APIError: If the request fails.
            ValueError: If the response is not valid JSON.
        """
        response = self.complete(prompt, system_prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Failed to parse response as JSON: {response[:200]}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_openai_vector_client() -> OpenAIVectorClient:
    """Create an OpenAI vector client using OPENAI_API_KEY env var.

    Returns:
        Configured OpenAIVectorClient instance.
    """
    api_key = os.environ.get('OPENAI_API_KEY', '')
    return OpenAIVectorClient(api_key=api_key)


def create_openai_chat_client() -> OpenAIChatClient:
    """Create an OpenAI chat client using OPENAI_API_KEY env var.

    Returns:
        Configured OpenAIChatClient instance.
    """
    api_key = os.environ.get('OPENAI_API_KEY', '')
    return OpenAIChatClient(api_key=api_key)


def create_vector_client() -> VectorAPIClient:
    """Create a vector API client using config settings.

    Returns:
        Configured VectorAPIClient instance.
    """
    return VectorAPIClient(
        endpoint=config.VECTOR_API_ENDPOINT,
        auth_header=config.VECTOR_API_AUTH_HEADER,
        auth_value=config.VECTOR_API_AUTH_VALUE,
        ssl_verify=config.SSL_VERIFY,
        ssl_cert_path=config.SSL_CERT_PATH
    )


def create_chat_client() -> ChatAPIClient:
    """Create a chat API client using config settings.

    Returns:
        Configured ChatAPIClient instance.
    """
    return ChatAPIClient(
        endpoint=config.CHAT_API_ENDPOINT,
        auth_header=config.CHAT_API_AUTH_HEADER,
        auth_value=config.CHAT_API_AUTH_VALUE,
        ssl_verify=config.SSL_VERIFY,
        ssl_cert_path=config.SSL_CERT_PATH
    )
