"""API client module for the RAG system.

Provides wrappers for vector embedding and chat completion APIs.
Includes OpenAI-compatible clients for use with OpenAI's API.
Uses only Python standard library (urllib).
Includes circuit breaker pattern for fault tolerance.
"""

import json
import os
import re
import ssl
import time
import threading
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any, Union, Callable, TypeVar
from functools import wraps

from rag_system import config

# Type variable for generic decorator
T = TypeVar('T')


# =============================================================================
# Custom Exceptions
# =============================================================================

class APIError(Exception):
    """Exception raised when an API call fails."""

    def __init__(self, message: str, status_code: Optional[int] = None,
                 response: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class CircuitBreakerError(APIError):
    """Exception raised when circuit breaker is open."""

    def __init__(self, message: str = "Circuit breaker is open"):
        super().__init__(message)


class TimeoutError(APIError):
    """Exception raised when API request times out."""

    def __init__(self, message: str = "API request timed out"):
        super().__init__(message)


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance.

    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, requests blocked immediately
    - HALF_OPEN: Testing, allowing a single request through

    After failure_threshold consecutive failures, the circuit opens.
    After reset_timeout seconds, the circuit enters half-open state.
    A successful request in half-open state closes the circuit.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0,
                 name: str = "default"):
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening.
            reset_timeout: Seconds to wait before half-open state.
            name: Name for logging/identification.
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name

        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._lock:
            if self._state == self.OPEN:
                # Check if we should transition to half-open
                if (self._last_failure_time and
                    time.time() - self._last_failure_time >= self.reset_timeout):
                    self._state = self.HALF_OPEN
            return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._failure_count = 0
            self._state = self.CLOSED

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = self.OPEN

    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        current_state = self.state
        return current_state in (self.CLOSED, self.HALF_OPEN)

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._last_failure_time = None


def with_circuit_breaker(circuit_breaker: CircuitBreaker) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to wrap a function with circuit breaker protection.

    Args:
        circuit_breaker: CircuitBreaker instance to use.

    Returns:
        Decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not circuit_breaker.can_execute():
                raise CircuitBreakerError(
                    f"Circuit breaker '{circuit_breaker.name}' is open - "
                    f"API unavailable after {circuit_breaker.failure_threshold} failures"
                )
            try:
                result = func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
            except Exception as e:
                circuit_breaker.record_failure()
                raise
        return wrapper
    return decorator


# Global circuit breakers for API clients
_vector_api_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    reset_timeout=60.0,
    name="vector_api"
)

_chat_api_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    reset_timeout=60.0,
    name="chat_api"
)


def get_vector_circuit_breaker() -> CircuitBreaker:
    """Get the global vector API circuit breaker."""
    return _vector_api_circuit_breaker


def get_chat_circuit_breaker() -> CircuitBreaker:
    """Get the global chat API circuit breaker."""
    return _chat_api_circuit_breaker


# =============================================================================
# Shared HTTP Utilities
# =============================================================================

# Default timeout for API requests (seconds)
DEFAULT_TIMEOUT: float = 30.0


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
                      ssl_context: Optional[ssl.SSLContext] = None,
                      max_retries: int = 3,
                      timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Make an HTTP POST request and return parsed JSON.

    Automatically retries on rate limit (429) and server error (5xx) codes
    with exponential backoff.

    Args:
        endpoint: API endpoint URL.
        data: Request body data.
        headers: HTTP headers.
        ssl_context: Optional SSL context.
        max_retries: Maximum number of retries for transient errors.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response.

    Raises:
        APIError: If the request fails after all retries.
        TimeoutError: If the request times out.
    """
    body = json.dumps(data).encode('utf-8')

    # Retryable status codes
    retryable_codes = {429, 500, 502, 503, 504}

    for attempt in range(max_retries + 1):
        request = urllib.request.Request(endpoint, data=body, headers=headers, method='POST')

        try:
            with urllib.request.urlopen(request, context=ssl_context, timeout=timeout) as response:
                response_data = response.read().decode('utf-8')
                try:
                    return json.loads(response_data)
                except json.JSONDecodeError as e:
                    raise APIError(
                        f"Failed to parse API response as JSON: {e}",
                        response=response_data[:500] if response_data else None
                    )
        except urllib.error.HTTPError as e:
            # Check if this is a retryable error
            if e.code in retryable_codes and attempt < max_retries:
                # Try to get Retry-After header, default to exponential backoff
                retry_after = e.headers.get('Retry-After') if e.headers else None
                if retry_after:
                    try:
                        delay = int(retry_after)
                    except ValueError:
                        delay = 2 ** attempt
                else:
                    delay = 2 ** attempt
                time.sleep(delay)
                continue

            # Read error response safely
            error_response = None
            try:
                if hasattr(e, 'read') and callable(e.read):
                    error_response = e.read().decode('utf-8')
            except Exception:
                pass  # Ignore errors reading the error response

            raise APIError(
                f"API request failed: {e.reason}",
                status_code=e.code,
                response=error_response
            )
        except urllib.error.URLError as e:
            # Check if it's a timeout
            if 'timed out' in str(e.reason).lower():
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise TimeoutError(f"API request timed out after {timeout}s")

            # Other URL errors (network issues, etc.)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue

            raise APIError(f"API request failed: {e.reason}")
        except Exception as e:
            # Catch socket timeout and other unexpected errors
            if 'timed out' in str(e).lower():
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise TimeoutError(f"API request timed out after {timeout}s")
            raise APIError(f"Unexpected error during API request: {e}")

    # Should not reach here, but satisfy type checker
    raise APIError("API request failed after all retries")


class VectorAPIClient:
    """Client for vector embedding API.

    Uses circuit breaker pattern to prevent cascading failures.
    """

    def __init__(self, endpoint: str, auth_header: str, auth_value: str,
                 ssl_verify: bool = True, ssl_cert_path: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT,
                 circuit_breaker: Optional[CircuitBreaker] = None):
        """Initialize the vector API client.

        Args:
            endpoint: API endpoint URL.
            auth_header: Name of the authentication header.
            auth_value: Value for the authentication header.
            ssl_verify: Whether to verify SSL certificates.
            ssl_cert_path: Path to SSL certificate file.
            timeout: Request timeout in seconds.
            circuit_breaker: Optional circuit breaker (uses global if not provided).
        """
        self.endpoint = endpoint
        self.auth_header = auth_header
        self.auth_value = auth_value
        self.ssl_context = create_ssl_context(ssl_verify, ssl_cert_path)
        self.timeout = timeout
        self._circuit_breaker = circuit_breaker or get_vector_circuit_breaker()

    def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP request to the API.

        Raises:
            CircuitBreakerError: If circuit breaker is open.
            APIError: If request fails.
        """
        if not self._circuit_breaker.can_execute():
            raise CircuitBreakerError(
                f"Vector API circuit breaker is open - "
                f"service unavailable after consecutive failures"
            )

        headers = {
            'Content-Type': 'application/json',
            self.auth_header: self.auth_value
        }

        try:
            result = make_http_request(
                self.endpoint, data, headers, self.ssl_context, timeout=self.timeout
            )
            self._circuit_breaker.record_success()
            return result
        except APIError:
            self._circuit_breaker.record_failure()
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            APIError: If the request fails.
            CircuitBreakerError: If circuit breaker is open.
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
            CircuitBreakerError: If circuit breaker is open.
        """
        response = self._make_request({'texts': texts})
        return response.get('embeddings', [])

    def is_available(self) -> bool:
        """Check if the API is available (circuit breaker closed/half-open)."""
        return self._circuit_breaker.can_execute()

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker to allow requests again."""
        self._circuit_breaker.reset()


class ChatAPIClient:
    """Client for chat completion API.

    Uses circuit breaker pattern to prevent cascading failures.
    """

    def __init__(self, endpoint: str, auth_header: str, auth_value: str,
                 ssl_verify: bool = True, ssl_cert_path: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT,
                 circuit_breaker: Optional[CircuitBreaker] = None):
        """Initialize the chat API client.

        Args:
            endpoint: API endpoint URL.
            auth_header: Name of the authentication header.
            auth_value: Value for the authentication header.
            ssl_verify: Whether to verify SSL certificates.
            ssl_cert_path: Path to SSL certificate file.
            timeout: Request timeout in seconds.
            circuit_breaker: Optional circuit breaker (uses global if not provided).
        """
        self.endpoint = endpoint
        self.auth_header = auth_header
        self.auth_value = auth_value
        self.ssl_context = create_ssl_context(ssl_verify, ssl_cert_path)
        self.timeout = timeout
        self._circuit_breaker = circuit_breaker or get_chat_circuit_breaker()

    def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP request to the API.

        Raises:
            CircuitBreakerError: If circuit breaker is open.
            APIError: If request fails.
        """
        if not self._circuit_breaker.can_execute():
            raise CircuitBreakerError(
                f"Chat API circuit breaker is open - "
                f"service unavailable after consecutive failures"
            )

        headers = {
            'Content-Type': 'application/json',
            self.auth_header: self.auth_value
        }

        try:
            result = make_http_request(
                self.endpoint, data, headers, self.ssl_context, timeout=self.timeout
            )
            self._circuit_breaker.record_success()
            return result
        except APIError:
            self._circuit_breaker.record_failure()
            raise

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Get a chat completion.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt for context.

        Returns:
            Completion response as a string.

        Raises:
            APIError: If the request fails.
            CircuitBreakerError: If circuit breaker is open.
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
            CircuitBreakerError: If circuit breaker is open.
            ValueError: If the response is not valid JSON.
        """
        response = self.complete(prompt, system_prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response as JSON: {e}")

    def is_available(self) -> bool:
        """Check if the API is available (circuit breaker closed/half-open)."""
        return self._circuit_breaker.can_execute()

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker to allow requests again."""
        self._circuit_breaker.reset()


# =============================================================================
# OpenAI API Clients
# =============================================================================

class OpenAIVectorClient:
    """Client for OpenAI embeddings API.

    Uses circuit breaker pattern to prevent cascading failures.
    """

    ENDPOINT = 'https://api.openai.com/v1/embeddings'

    def __init__(self, api_key: str, model: str = 'text-embedding-3-small',
                 timeout: float = DEFAULT_TIMEOUT,
                 circuit_breaker: Optional[CircuitBreaker] = None):
        """Initialize the OpenAI vector client.

        Args:
            api_key: OpenAI API key.
            model: Embedding model to use.
            timeout: Request timeout in seconds.
            circuit_breaker: Optional circuit breaker (uses global if not provided).
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._circuit_breaker = circuit_breaker or get_vector_circuit_breaker()

    def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP request to OpenAI API.

        Raises:
            CircuitBreakerError: If circuit breaker is open.
            APIError: If request fails.
        """
        if not self._circuit_breaker.can_execute():
            raise CircuitBreakerError(
                f"Vector API circuit breaker is open - "
                f"service unavailable after consecutive failures"
            )

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        try:
            result = make_http_request(self.ENDPOINT, data, headers, timeout=self.timeout)
            self._circuit_breaker.record_success()
            return result
        except APIError:
            self._circuit_breaker.record_failure()
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            APIError: If the request fails.
            CircuitBreakerError: If circuit breaker is open.
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
            CircuitBreakerError: If circuit breaker is open.
        """
        data = {'input': texts, 'model': self.model}
        response = self._make_request(data)
        sorted_data = sorted(response['data'], key=lambda x: x['index'])
        return [item['embedding'] for item in sorted_data]

    def is_available(self) -> bool:
        """Check if the API is available (circuit breaker closed/half-open)."""
        return self._circuit_breaker.can_execute()

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker to allow requests again."""
        self._circuit_breaker.reset()


class OpenAIChatClient:
    """Client for OpenAI chat completions API.

    Uses circuit breaker pattern to prevent cascading failures.
    """

    ENDPOINT = 'https://api.openai.com/v1/chat/completions'

    def __init__(self, api_key: str, model: str = 'gpt-4o-mini',
                 timeout: float = DEFAULT_TIMEOUT,
                 circuit_breaker: Optional[CircuitBreaker] = None):
        """Initialize the OpenAI chat client.

        Args:
            api_key: OpenAI API key.
            model: Chat model to use.
            timeout: Request timeout in seconds.
            circuit_breaker: Optional circuit breaker (uses global if not provided).
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._circuit_breaker = circuit_breaker or get_chat_circuit_breaker()

    def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP request to OpenAI API.

        Raises:
            CircuitBreakerError: If circuit breaker is open.
            APIError: If request fails.
        """
        if not self._circuit_breaker.can_execute():
            raise CircuitBreakerError(
                f"Chat API circuit breaker is open - "
                f"service unavailable after consecutive failures"
            )

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        try:
            result = make_http_request(self.ENDPOINT, data, headers, timeout=self.timeout)
            self._circuit_breaker.record_success()
            return result
        except APIError:
            self._circuit_breaker.record_failure()
            raise

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Get a chat completion.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt for context.

        Returns:
            Completion response as a string.

        Raises:
            APIError: If the request fails.
            CircuitBreakerError: If circuit breaker is open.
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
            CircuitBreakerError: If circuit breaker is open.
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

    def is_available(self) -> bool:
        """Check if the API is available (circuit breaker closed/half-open)."""
        return self._circuit_breaker.can_execute()

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker to allow requests again."""
        self._circuit_breaker.reset()


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
