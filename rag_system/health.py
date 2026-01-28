"""Health check module for the RAG system.

Provides health check functionality to verify system dependencies.
"""

from typing import Dict, List, Optional, Any
import sqlite3
import os

from rag_system import config
from rag_system.utils import get_logger

logger = get_logger(__name__)


class HealthCheckResult:
    """Result of a health check."""

    def __init__(self, name: str, healthy: bool, message: str = "",
                 details: Optional[Dict[str, Any]] = None):
        """Initialize a health check result.

        Args:
            name: Name of the component checked.
            healthy: Whether the component is healthy.
            message: Human-readable status message.
            details: Optional additional details.
        """
        self.name = name
        self.healthy = healthy
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'healthy': self.healthy,
            'message': self.message,
            'details': self.details
        }


class HealthChecker:
    """Performs health checks on system components."""

    def __init__(self, db_path: str = None,
                 vector_client: Any = None,
                 chat_client: Any = None):
        """Initialize the health checker.

        Args:
            db_path: Path to SQLite database.
            vector_client: Optional vector API client.
            chat_client: Optional chat API client.
        """
        self.db_path = db_path or config.DATABASE_PATH
        self.vector_client = vector_client
        self.chat_client = chat_client

    def check_database(self) -> HealthCheckResult:
        """Check database connectivity.

        Returns:
            HealthCheckResult for database.
        """
        try:
            # Check if database file exists (for existing databases)
            db_exists = os.path.exists(self.db_path)

            # Try to connect and execute a simple query
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.execute("SELECT sqlite_version()")
                version = cursor.fetchone()[0]

                # Get some stats
                cursor = conn.execute("SELECT COUNT(*) FROM pages")
                page_count = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM chunks")
                chunk_count = cursor.fetchone()[0]

                return HealthCheckResult(
                    name="database",
                    healthy=True,
                    message=f"Connected (SQLite {version})",
                    details={
                        'path': self.db_path,
                        'exists': db_exists,
                        'pages': page_count,
                        'chunks': chunk_count
                    }
                )
            finally:
                conn.close()

        except sqlite3.Error as e:
            return HealthCheckResult(
                name="database",
                healthy=False,
                message=f"Connection failed: {e}",
                details={'path': self.db_path}
            )

    def check_vector_api(self) -> HealthCheckResult:
        """Check Vector API connectivity.

        Returns:
            HealthCheckResult for Vector API.
        """
        if self.vector_client is None:
            return HealthCheckResult(
                name="vector_api",
                healthy=True,
                message="Not configured (optional)",
                details={'configured': False}
            )

        try:
            # Check if client has is_available method (circuit breaker)
            if hasattr(self.vector_client, 'is_available'):
                if not self.vector_client.is_available():
                    return HealthCheckResult(
                        name="vector_api",
                        healthy=False,
                        message="Circuit breaker is open",
                        details={'configured': True, 'circuit_breaker': 'open'}
                    )

            # Try a simple embedding request
            test_text = "health check"
            embedding = self.vector_client.get_embedding(test_text)

            if not embedding or not isinstance(embedding, list):
                return HealthCheckResult(
                    name="vector_api",
                    healthy=False,
                    message="Invalid response from API",
                    details={'configured': True}
                )

            return HealthCheckResult(
                name="vector_api",
                healthy=True,
                message=f"Connected (dimension: {len(embedding)})",
                details={
                    'configured': True,
                    'embedding_dimension': len(embedding)
                }
            )

        except Exception as e:
            return HealthCheckResult(
                name="vector_api",
                healthy=False,
                message=f"Connection failed: {e}",
                details={'configured': True}
            )

    def check_chat_api(self) -> HealthCheckResult:
        """Check Chat API connectivity.

        Returns:
            HealthCheckResult for Chat API.
        """
        if self.chat_client is None:
            return HealthCheckResult(
                name="chat_api",
                healthy=True,
                message="Not configured (optional)",
                details={'configured': False}
            )

        try:
            # Check if client has is_available method (circuit breaker)
            if hasattr(self.chat_client, 'is_available'):
                if not self.chat_client.is_available():
                    return HealthCheckResult(
                        name="chat_api",
                        healthy=False,
                        message="Circuit breaker is open",
                        details={'configured': True, 'circuit_breaker': 'open'}
                    )

            # Try a simple completion request
            test_prompt = "Say 'OK' if you can read this."
            response = self.chat_client.complete(test_prompt)

            if not response or not isinstance(response, str):
                return HealthCheckResult(
                    name="chat_api",
                    healthy=False,
                    message="Invalid response from API",
                    details={'configured': True}
                )

            return HealthCheckResult(
                name="chat_api",
                healthy=True,
                message="Connected",
                details={'configured': True}
            )

        except Exception as e:
            return HealthCheckResult(
                name="chat_api",
                healthy=False,
                message=f"Connection failed: {e}",
                details={'configured': True}
            )

    def check_configuration(self) -> HealthCheckResult:
        """Check configuration validity.

        Returns:
            HealthCheckResult for configuration.
        """
        try:
            from rag_system.config import validate_config

            is_valid, errors = validate_config(require_apis=False)

            if is_valid:
                return HealthCheckResult(
                    name="configuration",
                    healthy=True,
                    message="Valid",
                    details={'errors': []}
                )
            else:
                return HealthCheckResult(
                    name="configuration",
                    healthy=False,
                    message=f"{len(errors)} validation error(s)",
                    details={'errors': errors}
                )

        except Exception as e:
            return HealthCheckResult(
                name="configuration",
                healthy=False,
                message=f"Validation failed: {e}",
                details={}
            )

    def check_all(self) -> Dict[str, Any]:
        """Run all health checks.

        Returns:
            Dict with overall health status and individual check results.
        """
        results = [
            self.check_configuration(),
            self.check_database(),
            self.check_vector_api(),
            self.check_chat_api(),
        ]

        overall_healthy = all(r.healthy for r in results)

        return {
            'healthy': overall_healthy,
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'checks': [r.to_dict() for r in results]
        }


def format_health_report(health_result: Dict[str, Any]) -> str:
    """Format health check result for display.

    Args:
        health_result: Result from HealthChecker.check_all()

    Returns:
        Formatted string for CLI output.
    """
    lines = []

    status = health_result['status'].upper()
    status_symbol = "[OK]" if health_result['healthy'] else "[FAIL]"
    lines.append(f"RAG System Health: {status_symbol} {status}")
    lines.append("=" * 40)

    for check in health_result['checks']:
        symbol = "[OK]" if check['healthy'] else "[FAIL]"
        name = check['name'].replace('_', ' ').title()
        message = check['message']
        lines.append(f"  {symbol} {name}: {message}")

        # Show details for failed checks
        if not check['healthy'] and check.get('details'):
            details = check['details']
            if details.get('errors'):
                for error in details['errors'][:3]:  # Limit to 3 errors
                    lines.append(f"      - {error}")

    return '\n'.join(lines)


def check_api_availability(vector_client: Any = None, chat_client: Any = None,
                           require_vector: bool = False, require_chat: bool = False) -> bool:
    """Check if required APIs are available.

    Args:
        vector_client: Vector API client (optional).
        chat_client: Chat API client (optional).
        require_vector: Whether vector API is required.
        require_chat: Whether chat API is required.

    Returns:
        True if all required APIs are available.
    """
    if require_vector:
        if vector_client is None:
            logger.error("Vector API is required but not configured")
            return False

        if hasattr(vector_client, 'is_available') and not vector_client.is_available():
            logger.error("Vector API circuit breaker is open")
            return False

    if require_chat:
        if chat_client is None:
            logger.error("Chat API is required but not configured")
            return False

        if hasattr(chat_client, 'is_available') and not chat_client.is_available():
            logger.error("Chat API circuit breaker is open")
            return False

    return True
