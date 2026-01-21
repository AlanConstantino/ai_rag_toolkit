"""API client module for the RAG system.

Provides wrappers for vector embedding and chat completion APIs.
Uses only Python standard library (urllib).
"""

import json
import ssl
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any

from rag_system import config


class APIError(Exception):
    """Exception raised when an API call fails."""

    def __init__(self, message: str, status_code: Optional[int] = None,
                 response: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


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
        self.ssl_context = self._create_ssl_context(ssl_verify, ssl_cert_path)

    def _create_ssl_context(self, ssl_verify: bool,
                           ssl_cert_path: Optional[str]) -> Optional[ssl.SSLContext]:
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

    def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP request to the API.

        Args:
            data: Request body data.

        Returns:
            Parsed JSON response.

        Raises:
            APIError: If the request fails.
        """
        body = json.dumps(data).encode('utf-8')

        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                'Content-Type': 'application/json',
                self.auth_header: self.auth_value
            },
            method='POST'
        )

        try:
            with urllib.request.urlopen(request, context=self.ssl_context) as response:
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
        self.ssl_context = self._create_ssl_context(ssl_verify, ssl_cert_path)

    def _create_ssl_context(self, ssl_verify: bool,
                           ssl_cert_path: Optional[str]) -> Optional[ssl.SSLContext]:
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

    def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP request to the API.

        Args:
            data: Request body data.

        Returns:
            Parsed JSON response.

        Raises:
            APIError: If the request fails.
        """
        body = json.dumps(data).encode('utf-8')

        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                'Content-Type': 'application/json',
                self.auth_header: self.auth_value
            },
            method='POST'
        )

        try:
            with urllib.request.urlopen(request, context=self.ssl_context) as response:
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
# Factory Functions
# =============================================================================

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
