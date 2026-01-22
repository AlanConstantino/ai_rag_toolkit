"""Configuration settings for the RAG system.

All settings can be overridden via environment variables with RAG_ prefix.
Includes validation to catch misconfigurations at startup.
"""

import os
from typing import List, Optional, Tuple


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass

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
# Embedding Settings
# =============================================================================

EMBEDDING_BATCH_SIZE: int = int(os.environ.get('RAG_EMBEDDING_BATCH_SIZE', '100'))
EMBEDDING_BATCH_DELAY: float = float(os.environ.get('RAG_EMBEDDING_BATCH_DELAY', '0.1'))
EMBEDDING_MAX_RETRIES: int = int(os.environ.get('RAG_EMBEDDING_MAX_RETRIES', '3'))

# =============================================================================
# HTTP Cache Configuration
# =============================================================================

HTTP_CACHE_DIR: Optional[str] = os.environ.get('RAG_HTTP_CACHE_DIR', None)


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
