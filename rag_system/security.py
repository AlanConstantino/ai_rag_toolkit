"""Security utilities for the RAG system.

Provides input validation, content sanitization, and security checks.
"""

import html
import re
from typing import Optional, Tuple

# Maximum query length in characters
MAX_QUERY_LENGTH = 10000

# Maximum content length for database storage (1MB in characters)
MAX_CONTENT_LENGTH = 1000000

# Patterns that might indicate malicious content
SUSPICIOUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # Script tags
    r'javascript:',  # JavaScript protocol
    r'on\w+\s*=',  # Event handlers
    r'data:text/html',  # Data URLs with HTML
]


class ValidationError(Exception):
    """Exception raised for input validation failures."""

    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(f"Validation error for {field}: {message}")


def validate_query_length(query: str, max_length: int = MAX_QUERY_LENGTH) -> Tuple[bool, Optional[str]]:
    """Validate that a query is within the allowed length.

    Args:
        query: Query string to validate.
        max_length: Maximum allowed length.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not query:
        return False, "Query cannot be empty"

    if len(query) > max_length:
        return False, f"Query exceeds maximum length of {max_length} characters"

    return True, None


def validate_query_length_or_raise(query: str, max_length: int = MAX_QUERY_LENGTH) -> None:
    """Validate query length and raise ValidationError if invalid.

    Args:
        query: Query string to validate.
        max_length: Maximum allowed length.

    Raises:
        ValidationError: If query is invalid.
    """
    is_valid, error = validate_query_length(query, max_length)
    if not is_valid:
        raise ValidationError("query", error)


def validate_content_length(content: str, max_length: int = MAX_CONTENT_LENGTH) -> Tuple[bool, Optional[str]]:
    """Validate that content is within the allowed length.

    Args:
        content: Content string to validate.
        max_length: Maximum allowed length.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if content is None:
        return True, None  # None content is allowed (will be stored as empty)

    if len(content) > max_length:
        return False, f"Content exceeds maximum length of {max_length} characters"

    return True, None


def sanitize_html_content(content: str) -> str:
    """Sanitize HTML content to remove potentially dangerous elements.

    This removes script tags, event handlers, and other potentially
    dangerous HTML while preserving the text content.

    Args:
        content: HTML content to sanitize.

    Returns:
        Sanitized content.
    """
    if not content:
        return content

    # Remove script tags and their content
    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)

    # Remove style tags and their content
    content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.IGNORECASE | re.DOTALL)

    # Remove event handlers (onclick, onload, etc.)
    content = re.sub(r'\s+on\w+\s*=\s*["\'][^"\']*["\']', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\s+on\w+\s*=\s*\S+', '', content, flags=re.IGNORECASE)

    # Remove javascript: protocol
    content = re.sub(r'javascript:', '', content, flags=re.IGNORECASE)

    # Remove data: URLs that could contain executable content
    content = re.sub(r'data:text/html[^"\'>\s]*', '', content, flags=re.IGNORECASE)

    return content


def sanitize_for_logging(value: str, max_length: int = 200) -> str:
    """Sanitize a value for safe logging.

    Truncates long values and removes potentially sensitive patterns.

    Args:
        value: Value to sanitize.
        max_length: Maximum length for logged value.

    Returns:
        Sanitized value safe for logging.
    """
    if not value:
        return value

    # Truncate long values
    if len(value) > max_length:
        value = value[:max_length] + "...[truncated]"

    # Remove potential credential patterns
    value = re.sub(r'(password|secret|token|key|auth)\s*[:=]\s*\S+', r'\1=[REDACTED]',
                   value, flags=re.IGNORECASE)

    # Remove potential API keys (long alphanumeric strings)
    value = re.sub(r'[A-Za-z0-9]{32,}', '[REDACTED]', value)

    return value


def check_for_suspicious_content(content: str) -> Tuple[bool, Optional[str]]:
    """Check content for suspicious patterns.

    Args:
        content: Content to check.

    Returns:
        Tuple of (has_suspicious, pattern_description).
    """
    if not content:
        return False, None

    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            return True, f"Content contains suspicious pattern: {pattern}"

    return False, None
