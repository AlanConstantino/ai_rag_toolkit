"""Utility functions for the RAG system.

Provides logging, JSON parsing, text processing, and other helper functions.
Uses only Python standard library.
"""

import hashlib
import json
import logging
import re
import time
from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar


# =============================================================================
# Logging Utilities
# =============================================================================

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger.

    Args:
        name: Logger name (typically module name).
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


# =============================================================================
# JSON Parsing Utilities
# =============================================================================

def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON from text, with fallback extraction.

    Attempts to parse the text as JSON. If that fails, tries to extract
    JSON from within the text (e.g., from markdown code blocks or
    surrounding text).

    Args:
        text: Text potentially containing JSON.
        default: Value to return if no JSON can be extracted.

    Returns:
        Parsed JSON value, or default if parsing fails.
    """
    if default is None:
        default = {}

    # First try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try to find JSON object or array in text
    # Look for {...} or [...]
    json_patterns = [
        r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',  # Nested objects
        r'(\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])',  # Nested arrays
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    return default


# =============================================================================
# Hashing Utilities
# =============================================================================

def hash_content(content: str) -> str:
    """Generate a SHA-256 hash of content.

    Args:
        content: Text content to hash.

    Returns:
        Hex string of the hash.
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def get_query_hash(query: str) -> str:
    """Generate a normalized hash for a query.

    Normalizes the query (lowercase, strip whitespace) before hashing
    to ensure equivalent queries produce the same hash.

    Args:
        query: Query string.

    Returns:
        Hash string.
    """
    normalized = query.strip().lower()
    return hash_content(normalized)


# =============================================================================
# Text Processing Utilities
# =============================================================================

def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words.

    Removes punctuation and splits on whitespace.

    Args:
        text: Text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    if not text:
        return []

    # Remove punctuation except for numbers
    cleaned = re.sub(r"[^\w\s]", '', text)
    # Split on whitespace and lowercase
    tokens = cleaned.lower().split()
    return tokens


def clean_text(text: str) -> str:
    """Clean and normalize text.

    Removes extra whitespace and normalizes line breaks.

    Args:
        text: Text to clean.

    Returns:
        Cleaned text.
    """
    # Replace multiple whitespace/newlines with single space
    cleaned = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return cleaned.strip()


def truncate_text(text: str, max_length: int, suffix: str = '...') -> str:
    """Truncate text to a maximum length.

    Args:
        text: Text to truncate.
        max_length: Maximum length including suffix.
        suffix: Suffix to add when truncating.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text

    truncate_at = max_length - len(suffix)
    return text[:truncate_at] + suffix


# =============================================================================
# Timing Utilities
# =============================================================================

class Timer:
    """Context manager for timing code execution."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> 'Timer':
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time


# =============================================================================
# Retry Utilities
# =============================================================================

T = TypeVar('T')


def retry(max_attempts: int = 3, delay: float = 1.0,
          exceptions: tuple = (Exception,)) -> Callable:
    """Decorator to retry a function on failure.

    Args:
        max_attempts: Maximum number of attempts.
        delay: Delay between attempts in seconds.
        exceptions: Tuple of exceptions to catch and retry on.

    Returns:
        Decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator
