"""Utility functions for the RAG system.

Provides logging, JSON parsing, text processing, and other helper functions.
Uses only Python standard library.
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar


# =============================================================================
# Logging Utilities
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format.

        Returns:
            JSON string.
        """
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, 'extra_data') and record.extra_data:
            log_data['data'] = record.extra_data

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def get_log_level() -> int:
    """Get the configured log level.

    Returns:
        Logging level constant.
    """
    level_str = os.environ.get('RAG_LOG_LEVEL', 'INFO').upper()
    return getattr(logging, level_str, logging.INFO)


def get_log_format() -> str:
    """Get the configured log format.

    Returns:
        'json' or 'text'.
    """
    return os.environ.get('RAG_LOG_FORMAT', 'text').lower()


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a configured logger.

    Args:
        name: Logger name (typically module name).
        level: Logging level. If None, uses RAG_LOG_LEVEL env var.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()

        log_format = get_log_format()
        if log_format == 'json':
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if level is None:
        level = get_log_level()
    logger.setLevel(level)
    return logger


def log_with_context(logger: logging.Logger, level: int, message: str,
                     **kwargs: Any) -> None:
    """Log a message with additional context data.

    For JSON format, the kwargs become structured data fields.
    For text format, they're appended to the message.

    Args:
        logger: Logger instance.
        level: Logging level.
        message: Log message.
        **kwargs: Additional context data.
    """
    if get_log_format() == 'json':
        # Create a custom log record with extra data
        record = logger.makeRecord(
            logger.name, level, "", 0, message, (), None
        )
        record.extra_data = kwargs
        logger.handle(record)
    else:
        # Append context to message for text format
        if kwargs:
            context_str = ' '.join(f'{k}={v}' for k, v in kwargs.items())
            message = f"{message} [{context_str}]"
        logger.log(level, message)


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
# Metrics Collection
# =============================================================================

class MetricsCollector:
    """Collects and stores timing metrics for various operations.

    Thread-safe singleton for collecting metrics across the application.
    """

    _instance: Optional['MetricsCollector'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> 'MetricsCollector':
        """Singleton pattern to ensure one instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        if self._initialized:
            return
        self._initialized = True
        self._metrics: Dict[str, List[float]] = {}
        self._current_query_metrics: Dict[str, float] = {}
        self._metrics_lock = threading.Lock()

    def record(self, metric_name: str, value: float) -> None:
        """Record a metric value.

        Args:
            metric_name: Name of the metric (e.g., 'vector_api_time').
            value: Metric value (typically time in seconds).
        """
        with self._metrics_lock:
            if metric_name not in self._metrics:
                self._metrics[metric_name] = []
            self._metrics[metric_name].append(value)

    def start_query(self) -> None:
        """Start tracking metrics for a new query."""
        with self._metrics_lock:
            self._current_query_metrics = {
                'start_time': time.time()
            }

    def record_query_metric(self, metric_name: str, value: float) -> None:
        """Record a metric for the current query.

        Args:
            metric_name: Name of the metric.
            value: Metric value.
        """
        with self._metrics_lock:
            self._current_query_metrics[metric_name] = value

    def finish_query(self) -> Dict[str, float]:
        """Finish tracking the current query and return metrics.

        Returns:
            Dict of all metrics collected for this query.
        """
        with self._metrics_lock:
            if 'start_time' in self._current_query_metrics:
                self._current_query_metrics['total_time'] = (
                    time.time() - self._current_query_metrics['start_time']
                )
            metrics = self._current_query_metrics.copy()
            self._current_query_metrics = {}
            return metrics

    def get_current_query_metrics(self) -> Dict[str, float]:
        """Get metrics for the current query without clearing.

        Returns:
            Dict of current query metrics.
        """
        with self._metrics_lock:
            return self._current_query_metrics.copy()

    def get_aggregate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get aggregate statistics for all recorded metrics.

        Returns:
            Dict with min, max, avg, count for each metric.
        """
        with self._metrics_lock:
            result = {}
            for name, values in self._metrics.items():
                if values:
                    result[name] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'total': sum(values)
                    }
            return result

    def get_all_metrics(self) -> Dict[str, List[float]]:
        """Get all raw metric values.

        Returns:
            Dict mapping metric names to lists of values.
        """
        with self._metrics_lock:
            return {k: v.copy() for k, v in self._metrics.items()}

    def clear(self) -> None:
        """Clear all collected metrics."""
        with self._metrics_lock:
            self._metrics.clear()
            self._current_query_metrics.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as a dictionary.

        Returns:
            Dict containing aggregate and raw metrics.
        """
        return {
            'aggregate': self.get_aggregate_metrics(),
            'raw': self.get_all_metrics()
        }


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.

    Returns:
        MetricsCollector singleton instance.
    """
    return MetricsCollector()


def timed_operation(metric_name: str) -> Callable:
    """Decorator to time a function and record the metric.

    Args:
        metric_name: Name of the metric to record.

    Returns:
        Decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            collector = get_metrics_collector()
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                collector.record(metric_name, elapsed)
                collector.record_query_metric(metric_name, elapsed)
        return wrapper
    return decorator


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
