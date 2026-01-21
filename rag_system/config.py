"""Configuration settings for the RAG system.

All settings can be overridden via environment variables with RAG_ prefix.
"""

import os

# =============================================================================
# API Configuration
# =============================================================================

VECTOR_API_ENDPOINT = os.environ.get(
    'RAG_VECTOR_API_ENDPOINT',
    'https://your-vector-api.com/embed'
)
VECTOR_API_AUTH_HEADER = os.environ.get(
    'RAG_VECTOR_API_AUTH_HEADER',
    'Authorization'
)
VECTOR_API_AUTH_VALUE = os.environ.get(
    'RAG_VECTOR_API_AUTH_VALUE',
    'Bearer your-token'
)

CHAT_API_ENDPOINT = os.environ.get(
    'RAG_CHAT_API_ENDPOINT',
    'https://your-chat-api.com/complete'
)
CHAT_API_AUTH_HEADER = os.environ.get(
    'RAG_CHAT_API_AUTH_HEADER',
    'Authorization'
)
CHAT_API_AUTH_VALUE = os.environ.get(
    'RAG_CHAT_API_AUTH_VALUE',
    'Bearer your-token'
)

# =============================================================================
# SSL Configuration
# =============================================================================

SSL_CERT_PATH = os.environ.get('RAG_SSL_CERT_PATH', None)
SSL_VERIFY = os.environ.get('RAG_SSL_VERIFY', 'true').lower() in ('true', '1', 'yes')

# =============================================================================
# Database Configuration
# =============================================================================

DATABASE_PATH = os.environ.get('RAG_DATABASE_PATH', 'rag_system.db')

# =============================================================================
# Crawler Settings
# =============================================================================

CRAWL_DELAY_SECONDS = float(os.environ.get('RAG_CRAWL_DELAY_SECONDS', '1.0'))
MAX_PAGES = int(os.environ.get('RAG_MAX_PAGES', '1000'))

# Parse ALLOWED_DOMAINS from comma-separated string
_allowed_domains = os.environ.get('RAG_ALLOWED_DOMAINS', '')
ALLOWED_DOMAINS = [d.strip() for d in _allowed_domains.split(',') if d.strip()] if _allowed_domains else []

# Parse EXCLUDED_PATHS from comma-separated string
_excluded_paths = os.environ.get('RAG_EXCLUDED_PATHS', '/api/,/static/')
EXCLUDED_PATHS = [p.strip() for p in _excluded_paths.split(',') if p.strip()]

# Parse INCLUDED_PATHS from comma-separated string
_included_paths = os.environ.get('RAG_INCLUDED_PATHS', '')
INCLUDED_PATHS = [p.strip() for p in _included_paths.split(',') if p.strip()] if _included_paths else None

# =============================================================================
# Chunking Settings
# =============================================================================

SMALL_CHUNK_SIZE = int(os.environ.get('RAG_SMALL_CHUNK_SIZE', '500'))
LARGE_CHUNK_SIZE = int(os.environ.get('RAG_LARGE_CHUNK_SIZE', '2000'))
CHUNK_OVERLAP = int(os.environ.get('RAG_CHUNK_OVERLAP', '100'))

# =============================================================================
# Search Settings
# =============================================================================

BM25_K1 = float(os.environ.get('RAG_BM25_K1', '1.5'))
BM25_B = float(os.environ.get('RAG_BM25_B', '0.75'))
VECTOR_WEIGHT = float(os.environ.get('RAG_VECTOR_WEIGHT', '0.7'))
BM25_WEIGHT = float(os.environ.get('RAG_BM25_WEIGHT', '0.3'))
TOP_K_RETRIEVAL = int(os.environ.get('RAG_TOP_K_RETRIEVAL', '20'))
TOP_K_FINAL = int(os.environ.get('RAG_TOP_K_FINAL', '5'))
MAX_CHUNKS_PER_PAGE = int(os.environ.get('RAG_MAX_CHUNKS_PER_PAGE', '2'))

# =============================================================================
# Confidence Settings
# =============================================================================

MIN_CONFIDENCE_SCORE = int(os.environ.get('RAG_MIN_CONFIDENCE_SCORE', '3'))
CONFIDENCE_THRESHOLD = float(os.environ.get('RAG_CONFIDENCE_THRESHOLD', '0.5'))

# =============================================================================
# Embedding Settings
# =============================================================================

EMBEDDING_BATCH_SIZE = int(os.environ.get('RAG_EMBEDDING_BATCH_SIZE', '100'))
EMBEDDING_BATCH_DELAY = float(os.environ.get('RAG_EMBEDDING_BATCH_DELAY', '0.1'))
EMBEDDING_MAX_RETRIES = int(os.environ.get('RAG_EMBEDDING_MAX_RETRIES', '3'))

# =============================================================================
# HTTP Cache Configuration
# =============================================================================

HTTP_CACHE_DIR = os.environ.get('RAG_HTTP_CACHE_DIR', None)
