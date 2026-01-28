"""Configuration settings for the RAG system.

All settings can be overridden via environment variables with RAG_ prefix.
Includes validation to catch misconfigurations at startup.
"""

import base64
import os
from typing import List, Optional, Tuple


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass

# =============================================================================
# AI Feature Toggle
# =============================================================================

# When disabled, the system uses BM25 search only (no embeddings or LLM calls)
AI_ENABLED: bool = os.environ.get('RAG_AI_ENABLED', 'false').lower() in ('true', '1', 'yes')

# =============================================================================
# API Configuration
# =============================================================================

VECTOR_API_ENDPOINT: str = os.environ.get(
    'RAG_VECTOR_API_ENDPOINT',
    'https://your-vector-api.com/embed'
)
VECTOR_API_AUTH_HEADER: str = os.environ.get(
    'RAG_VECTOR_API_AUTH_HEADER',
    'Authorization'
)
VECTOR_API_AUTH_VALUE: str = os.environ.get(
    'RAG_VECTOR_API_AUTH_VALUE',
    'Bearer your-token'
)

CHAT_API_ENDPOINT: str = os.environ.get(
    'RAG_CHAT_API_ENDPOINT',
    'https://your-chat-api.com/complete'
)
CHAT_API_AUTH_HEADER: str = os.environ.get(
    'RAG_CHAT_API_AUTH_HEADER',
    'Authorization'
)
CHAT_API_AUTH_VALUE: str = os.environ.get(
    'RAG_CHAT_API_AUTH_VALUE',
    'Bearer your-token'
)

# =============================================================================
# SSL Configuration
# =============================================================================

SSL_CERT_PATH: Optional[str] = os.environ.get('RAG_SSL_CERT_PATH', None)
SSL_VERIFY: bool = os.environ.get('RAG_SSL_VERIFY', 'true').lower() in ('true', '1', 'yes')

# =============================================================================
# Database Configuration
# =============================================================================

DATABASE_PATH: str = os.environ.get('RAG_DATABASE_PATH', 'rag_system.db')

# =============================================================================
# Crawler Settings
# =============================================================================

CRAWL_DELAY_SECONDS: float = float(os.environ.get('RAG_CRAWL_DELAY_SECONDS', '1.0'))
MAX_PAGES: int = int(os.environ.get('RAG_MAX_PAGES', '1000'))

# Parse ALLOWED_DOMAINS from comma-separated string
_allowed_domains: str = os.environ.get('RAG_ALLOWED_DOMAINS', '')
ALLOWED_DOMAINS: List[str] = [d.strip() for d in _allowed_domains.split(',') if d.strip()] if _allowed_domains else []

# Parse EXCLUDED_PATHS from comma-separated string
_excluded_paths: str = os.environ.get('RAG_EXCLUDED_PATHS', '/api/,/static/')
EXCLUDED_PATHS: List[str] = [p.strip() for p in _excluded_paths.split(',') if p.strip()]

# Parse INCLUDED_PATHS from comma-separated string
_included_paths: str = os.environ.get('RAG_INCLUDED_PATHS', '')
INCLUDED_PATHS: Optional[List[str]] = [p.strip() for p in _included_paths.split(',') if p.strip()] if _included_paths else None

# Retry settings for transient errors
CRAWLER_MAX_RETRIES: int = int(os.environ.get('RAG_CRAWLER_MAX_RETRIES', '3'))
CRAWLER_RETRY_DELAY: float = float(os.environ.get('RAG_CRAWLER_RETRY_DELAY', '1.0'))
CRAWLER_RETRY_STATUS_CODES: List[int] = [
    int(code.strip())
    for code in os.environ.get('RAG_CRAWLER_RETRY_STATUS_CODES', '500,502,503,504').split(',')
    if code.strip()
]

# =============================================================================
# HTTP Basic Auth for Crawler
# =============================================================================

# Enable HTTP Basic Auth for crawling authenticated websites
BASIC_AUTH_ENABLED: bool = os.environ.get('RAG_BASIC_AUTH_ENABLED', 'false').lower() in ('true', '1', 'yes')

# Username for Basic Auth (used if BASIC_AUTH_TOKEN not provided)
BASIC_AUTH_USERNAME: Optional[str] = os.environ.get('RAG_BASIC_AUTH_USERNAME', None)

# Password for Basic Auth (used if BASIC_AUTH_TOKEN not provided)
BASIC_AUTH_PASSWORD: Optional[str] = os.environ.get('RAG_BASIC_AUTH_PASSWORD', None)

# Pre-encoded Base64 token (takes precedence over username:password)
# Format: base64(username:password)
BASIC_AUTH_TOKEN: Optional[str] = os.environ.get('RAG_BASIC_AUTH_TOKEN', None)

# =============================================================================
# Main Content Detection
# =============================================================================

# Selectors to identify main content areas (checked in order of priority)
MAIN_CONTENT_SELECTORS: List[str] = [
    'main',           # HTML5 main tag
    'article',        # HTML5 article tag
    'role=main',      # ARIA role
    'class=main-content',
    'class=article-content',
    'class=post-content',
    'class=entry-content',
    'class=page-content',
    'class=content-area',
    'id=main-content',
    'id=content',
    'id=main',
]

# =============================================================================
# Chunking Settings
# =============================================================================

# Character-based chunk sizes
SMALL_CHUNK_SIZE: int = int(os.environ.get('RAG_SMALL_CHUNK_SIZE', '500'))
LARGE_CHUNK_SIZE: int = int(os.environ.get('RAG_LARGE_CHUNK_SIZE', '2000'))
CHUNK_OVERLAP: int = int(os.environ.get('RAG_CHUNK_OVERLAP', '100'))

# Token-based chunk sizes
USE_TOKEN_CHUNKING: bool = os.environ.get('RAG_USE_TOKEN_CHUNKING', 'false').lower() in ('true', '1', 'yes')
SMALL_CHUNK_TOKENS: int = int(os.environ.get('RAG_SMALL_CHUNK_TOKENS', '128'))
LARGE_CHUNK_TOKENS: int = int(os.environ.get('RAG_LARGE_CHUNK_TOKENS', '512'))
CHUNK_OVERLAP_TOKENS: int = int(os.environ.get('RAG_CHUNK_OVERLAP_TOKENS', '25'))

# =============================================================================
# Search Settings
# =============================================================================

BM25_K1: float = float(os.environ.get('RAG_BM25_K1', '1.5'))
BM25_B: float = float(os.environ.get('RAG_BM25_B', '0.75'))
VECTOR_WEIGHT: float = float(os.environ.get('RAG_VECTOR_WEIGHT', '0.7'))
BM25_WEIGHT: float = float(os.environ.get('RAG_BM25_WEIGHT', '0.3'))
TOP_K_RETRIEVAL: int = int(os.environ.get('RAG_TOP_K_RETRIEVAL', '20'))
TOP_K_FINAL: int = int(os.environ.get('RAG_TOP_K_FINAL', '5'))
MAX_CHUNKS_PER_PAGE: int = int(os.environ.get('RAG_MAX_CHUNKS_PER_PAGE', '2'))

# =============================================================================
# Confidence Settings
# =============================================================================

MIN_CONFIDENCE_SCORE: int = int(os.environ.get('RAG_MIN_CONFIDENCE_SCORE', '3'))
CONFIDENCE_THRESHOLD: float = float(os.environ.get('RAG_CONFIDENCE_THRESHOLD', '0.5'))

# =============================================================================
# Entity Extraction Settings
# =============================================================================

# Enable entity extraction during indexing (requires chat client)
ENTITY_EXTRACTION_ENABLED: bool = os.environ.get(
    'RAG_ENTITY_EXTRACTION_ENABLED', 'false'
).lower() in ('true', '1', 'yes')

# =============================================================================
# Page Summarization Settings
# =============================================================================

# Enable page summarization during indexing (requires chat client)
PAGE_SUMMARIZATION_ENABLED: bool = os.environ.get(
    'RAG_PAGE_SUMMARIZATION_ENABLED', 'false'
).lower() in ('true', '1', 'yes')

# =============================================================================
# System Summary Settings
# =============================================================================

# Enable automatic system and global summary generation after crawl
SYSTEM_SUMMARIES_ENABLED: bool = os.environ.get(
    'RAG_SYSTEM_SUMMARIES_ENABLED', 'false'
).lower() in ('true', '1', 'yes')

# =============================================================================
# Embedding Settings
# =============================================================================

EMBEDDING_BATCH_SIZE: int = int(os.environ.get('RAG_EMBEDDING_BATCH_SIZE', '100'))
EMBEDDING_BATCH_DELAY: float = float(os.environ.get('RAG_EMBEDDING_BATCH_DELAY', '0.1'))
EMBEDDING_MAX_RETRIES: int = int(os.environ.get('RAG_EMBEDDING_MAX_RETRIES', '3'))
EMBEDDING_STOP_ON_RATE_LIMIT: bool = os.environ.get('RAG_EMBEDDING_STOP_ON_RATE_LIMIT', 'true').lower() in ('true', '1', 'yes')
EMBEDDING_RETRY_DELAY: float = float(os.environ.get('RAG_EMBEDDING_RETRY_DELAY', '1.0'))

# =============================================================================
# HTTP Cache Configuration
# =============================================================================

HTTP_CACHE_DIR: Optional[str] = os.environ.get('RAG_HTTP_CACHE_DIR', None)

# =============================================================================
# Logging and Metrics Configuration
# =============================================================================

# Log format: 'text' for human-readable, 'json' for structured
LOG_FORMAT: str = os.environ.get('RAG_LOG_FORMAT', 'text')
LOG_LEVEL: str = os.environ.get('RAG_LOG_LEVEL', 'INFO')

# Enable metrics collection
METRICS_ENABLED: bool = os.environ.get('RAG_METRICS_ENABLED', 'true').lower() in ('true', '1', 'yes')

# =============================================================================
# Query Cache Configuration
# =============================================================================

# Enable query result caching
QUERY_CACHE_ENABLED: bool = os.environ.get('RAG_QUERY_CACHE_ENABLED', 'true').lower() in ('true', '1', 'yes')

# Maximum number of query results to cache
QUERY_CACHE_SIZE: int = int(os.environ.get('RAG_QUERY_CACHE_SIZE', '100'))

# Time-to-live for cached results in seconds (default: 1 hour)
QUERY_CACHE_TTL: int = int(os.environ.get('RAG_QUERY_CACHE_TTL', '3600'))


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config(require_apis: bool = False) -> Tuple[bool, List[str]]:
    """Validate configuration settings.

    Args:
        require_apis: If True, require API endpoints to be configured
                      (not just using default placeholder values)

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors: List[str] = []

    # Check for placeholder API values if APIs are required
    if require_apis:
        if VECTOR_API_ENDPOINT == 'https://your-vector-api.com/embed':
            errors.append(
                "RAG_VECTOR_API_ENDPOINT not configured "
                "(still using placeholder value)"
            )
        if VECTOR_API_AUTH_VALUE == 'Bearer your-token':
            errors.append(
                "RAG_VECTOR_API_AUTH_VALUE not configured "
                "(still using placeholder value)"
            )
        if CHAT_API_ENDPOINT == 'https://your-chat-api.com/complete':
            errors.append(
                "RAG_CHAT_API_ENDPOINT not configured "
                "(still using placeholder value)"
            )
        if CHAT_API_AUTH_VALUE == 'Bearer your-token':
            errors.append(
                "RAG_CHAT_API_AUTH_VALUE not configured "
                "(still using placeholder value)"
            )

    # Validate numeric ranges
    if CRAWL_DELAY_SECONDS < 0:
        errors.append(
            f"RAG_CRAWL_DELAY_SECONDS must be non-negative, got {CRAWL_DELAY_SECONDS}"
        )

    if MAX_PAGES <= 0:
        errors.append(
            f"RAG_MAX_PAGES must be positive, got {MAX_PAGES}"
        )

    if SMALL_CHUNK_SIZE <= 0:
        errors.append(
            f"RAG_SMALL_CHUNK_SIZE must be positive, got {SMALL_CHUNK_SIZE}"
        )

    if LARGE_CHUNK_SIZE <= 0:
        errors.append(
            f"RAG_LARGE_CHUNK_SIZE must be positive, got {LARGE_CHUNK_SIZE}"
        )

    if SMALL_CHUNK_SIZE > LARGE_CHUNK_SIZE:
        errors.append(
            f"RAG_SMALL_CHUNK_SIZE ({SMALL_CHUNK_SIZE}) must be <= "
            f"RAG_LARGE_CHUNK_SIZE ({LARGE_CHUNK_SIZE})"
        )

    if CHUNK_OVERLAP < 0:
        errors.append(
            f"RAG_CHUNK_OVERLAP must be non-negative, got {CHUNK_OVERLAP}"
        )

    if CHUNK_OVERLAP >= SMALL_CHUNK_SIZE:
        errors.append(
            f"RAG_CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be < "
            f"RAG_SMALL_CHUNK_SIZE ({SMALL_CHUNK_SIZE})"
        )

    # Validate token-based chunking if enabled
    if USE_TOKEN_CHUNKING:
        if SMALL_CHUNK_TOKENS <= 0:
            errors.append(
                f"RAG_SMALL_CHUNK_TOKENS must be positive, got {SMALL_CHUNK_TOKENS}"
            )
        if LARGE_CHUNK_TOKENS <= 0:
            errors.append(
                f"RAG_LARGE_CHUNK_TOKENS must be positive, got {LARGE_CHUNK_TOKENS}"
            )
        if SMALL_CHUNK_TOKENS > LARGE_CHUNK_TOKENS:
            errors.append(
                f"RAG_SMALL_CHUNK_TOKENS ({SMALL_CHUNK_TOKENS}) must be <= "
                f"RAG_LARGE_CHUNK_TOKENS ({LARGE_CHUNK_TOKENS})"
            )
        if CHUNK_OVERLAP_TOKENS < 0:
            errors.append(
                f"RAG_CHUNK_OVERLAP_TOKENS must be non-negative, got {CHUNK_OVERLAP_TOKENS}"
            )
        if CHUNK_OVERLAP_TOKENS >= SMALL_CHUNK_TOKENS:
            errors.append(
                f"RAG_CHUNK_OVERLAP_TOKENS ({CHUNK_OVERLAP_TOKENS}) must be < "
                f"RAG_SMALL_CHUNK_TOKENS ({SMALL_CHUNK_TOKENS})"
            )

    # Validate BM25 parameters
    if BM25_K1 < 0:
        errors.append(
            f"RAG_BM25_K1 must be non-negative, got {BM25_K1}"
        )

    if not (0 <= BM25_B <= 1):
        errors.append(
            f"RAG_BM25_B must be between 0 and 1, got {BM25_B}"
        )

    # Validate search weights
    if VECTOR_WEIGHT < 0 or VECTOR_WEIGHT > 1:
        errors.append(
            f"RAG_VECTOR_WEIGHT must be between 0 and 1, got {VECTOR_WEIGHT}"
        )

    if BM25_WEIGHT < 0 or BM25_WEIGHT > 1:
        errors.append(
            f"RAG_BM25_WEIGHT must be between 0 and 1, got {BM25_WEIGHT}"
        )

    # Only validate weight sum when AI is enabled (hybrid search requires it)
    if AI_ENABLED:
        weight_sum = VECTOR_WEIGHT + BM25_WEIGHT
        if abs(weight_sum - 1.0) > 0.001:
            errors.append(
                f"RAG_VECTOR_WEIGHT + RAG_BM25_WEIGHT must equal 1.0, "
                f"got {VECTOR_WEIGHT} + {BM25_WEIGHT} = {weight_sum}"
            )

    # Validate retrieval settings
    if TOP_K_RETRIEVAL <= 0:
        errors.append(
            f"RAG_TOP_K_RETRIEVAL must be positive, got {TOP_K_RETRIEVAL}"
        )

    if TOP_K_FINAL <= 0:
        errors.append(
            f"RAG_TOP_K_FINAL must be positive, got {TOP_K_FINAL}"
        )

    if TOP_K_FINAL > TOP_K_RETRIEVAL:
        errors.append(
            f"RAG_TOP_K_FINAL ({TOP_K_FINAL}) must be <= "
            f"RAG_TOP_K_RETRIEVAL ({TOP_K_RETRIEVAL})"
        )

    if MAX_CHUNKS_PER_PAGE <= 0:
        errors.append(
            f"RAG_MAX_CHUNKS_PER_PAGE must be positive, got {MAX_CHUNKS_PER_PAGE}"
        )

    # Validate confidence settings
    if not (1 <= MIN_CONFIDENCE_SCORE <= 5):
        errors.append(
            f"RAG_MIN_CONFIDENCE_SCORE must be between 1 and 5, "
            f"got {MIN_CONFIDENCE_SCORE}"
        )

    if not (0 <= CONFIDENCE_THRESHOLD <= 1):
        errors.append(
            f"RAG_CONFIDENCE_THRESHOLD must be between 0 and 1, "
            f"got {CONFIDENCE_THRESHOLD}"
        )

    # Validate embedding settings
    if EMBEDDING_BATCH_SIZE <= 0:
        errors.append(
            f"RAG_EMBEDDING_BATCH_SIZE must be positive, got {EMBEDDING_BATCH_SIZE}"
        )

    if EMBEDDING_BATCH_DELAY < 0:
        errors.append(
            f"RAG_EMBEDDING_BATCH_DELAY must be non-negative, "
            f"got {EMBEDDING_BATCH_DELAY}"
        )

    if EMBEDDING_MAX_RETRIES < 0:
        errors.append(
            f"RAG_EMBEDDING_MAX_RETRIES must be non-negative, "
            f"got {EMBEDDING_MAX_RETRIES}"
        )

    if EMBEDDING_RETRY_DELAY < 0:
        errors.append(
            f"RAG_EMBEDDING_RETRY_DELAY must be non-negative, "
            f"got {EMBEDDING_RETRY_DELAY}"
        )

    # Validate crawler retry settings
    if CRAWLER_MAX_RETRIES < 0:
        errors.append(
            f"RAG_CRAWLER_MAX_RETRIES must be non-negative, got {CRAWLER_MAX_RETRIES}"
        )

    if CRAWLER_RETRY_DELAY < 0:
        errors.append(
            f"RAG_CRAWLER_RETRY_DELAY must be non-negative, got {CRAWLER_RETRY_DELAY}"
        )

    # Validate SSL cert path if specified
    if SSL_CERT_PATH and not os.path.exists(SSL_CERT_PATH):
        errors.append(
            f"RAG_SSL_CERT_PATH file does not exist: {SSL_CERT_PATH}"
        )

    # Validate HTTP cache directory if specified
    if HTTP_CACHE_DIR:
        cache_parent = os.path.dirname(HTTP_CACHE_DIR) or '.'
        if not os.path.exists(cache_parent):
            errors.append(
                f"RAG_HTTP_CACHE_DIR parent directory does not exist: {cache_parent}"
            )

    # Validate query cache settings
    if QUERY_CACHE_SIZE <= 0:
        errors.append(
            f"RAG_QUERY_CACHE_SIZE must be positive, got {QUERY_CACHE_SIZE}"
        )

    if QUERY_CACHE_TTL < 0:
        errors.append(
            f"RAG_QUERY_CACHE_TTL must be non-negative, got {QUERY_CACHE_TTL}"
        )

    # Validate Basic Auth settings
    if BASIC_AUTH_ENABLED:
        if not BASIC_AUTH_TOKEN:
            # If no pre-encoded token, both username and password are required
            if not BASIC_AUTH_USERNAME or not BASIC_AUTH_PASSWORD:
                errors.append(
                    "RAG_BASIC_AUTH_ENABLED is true but credentials are incomplete. "
                    "Provide either RAG_BASIC_AUTH_TOKEN or both "
                    "RAG_BASIC_AUTH_USERNAME and RAG_BASIC_AUTH_PASSWORD"
                )

    return (len(errors) == 0, errors)


def validate_or_raise(require_apis: bool = False) -> None:
    """Validate configuration and raise ConfigurationError if invalid.

    Args:
        require_apis: If True, require API endpoints to be configured

    Raises:
        ConfigurationError: If any validation errors are found
    """
    is_valid, errors = validate_config(require_apis)
    if not is_valid:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        raise ConfigurationError(error_msg)


def get_basic_auth_token(username: Optional[str] = None,
                         password: Optional[str] = None,
                         token: Optional[str] = None) -> Optional[str]:
    """Get the HTTP Basic Auth token for crawler authentication.

    If a pre-encoded token is provided, it is returned as-is.
    Otherwise, encodes username:password as base64.

    Args:
        username: Username (defaults to BASIC_AUTH_USERNAME config).
        password: Password (defaults to BASIC_AUTH_PASSWORD config).
        token: Pre-encoded base64 token (defaults to BASIC_AUTH_TOKEN config).

    Returns:
        Base64-encoded token string, or None if auth is not configured.
    """
    # Use provided values or fall back to config
    token = token or BASIC_AUTH_TOKEN
    username = username or BASIC_AUTH_USERNAME
    password = password or BASIC_AUTH_PASSWORD

    # Pre-encoded token takes precedence
    if token:
        return token

    # Encode username:password
    if username and password:
        credentials = f"{username}:{password}"
        return base64.b64encode(credentials.encode('utf-8')).decode('utf-8')

    return None
