"""Tests for the health check module."""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os

from rag_system.health import (
    HealthCheckResult,
    HealthChecker,
    format_health_report,
    check_api_availability,
)


class TestHealthCheckResult(unittest.TestCase):
    """Tests for HealthCheckResult class."""

    def test_healthy_result(self):
        """HealthCheckResult should store healthy state."""
        result = HealthCheckResult(
            name="test",
            healthy=True,
            message="All good"
        )

        self.assertTrue(result.healthy)
        self.assertEqual(result.name, "test")
        self.assertEqual(result.message, "All good")

    def test_unhealthy_result(self):
        """HealthCheckResult should store unhealthy state."""
        result = HealthCheckResult(
            name="test",
            healthy=False,
            message="Failed",
            details={'error': 'something went wrong'}
        )

        self.assertFalse(result.healthy)
        self.assertEqual(result.details['error'], 'something went wrong')

    def test_to_dict(self):
        """to_dict should return dictionary representation."""
        result = HealthCheckResult(
            name="database",
            healthy=True,
            message="Connected",
            details={'version': '3.36.0'}
        )

        d = result.to_dict()

        self.assertEqual(d['name'], 'database')
        self.assertTrue(d['healthy'])
        self.assertEqual(d['message'], 'Connected')
        self.assertEqual(d['details']['version'], '3.36.0')


class TestHealthChecker(unittest.TestCase):
    """Tests for HealthChecker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_file.name
        self.temp_file.close()

        # Initialize database
        from rag_system.database import init_db
        init_db(self.db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.db_path)

    def test_check_database_healthy(self):
        """check_database should return healthy for valid database."""
        checker = HealthChecker(db_path=self.db_path)

        result = checker.check_database()

        self.assertTrue(result.healthy)
        self.assertEqual(result.name, "database")
        self.assertIn("SQLite", result.message)

    def test_check_database_unhealthy(self):
        """check_database should return unhealthy for invalid database."""
        checker = HealthChecker(db_path="/nonexistent/path/db.sqlite")

        result = checker.check_database()

        # SQLite will create the file, but we want to test with a truly bad path
        # Let's use a directory path which will fail
        checker = HealthChecker(db_path="/")

        result = checker.check_database()

        self.assertFalse(result.healthy)

    def test_check_vector_api_not_configured(self):
        """check_vector_api should be healthy when not configured."""
        checker = HealthChecker(db_path=self.db_path, vector_client=None)

        result = checker.check_vector_api()

        self.assertTrue(result.healthy)
        self.assertIn("Not configured", result.message)

    def test_check_vector_api_healthy(self):
        """check_vector_api should be healthy when API responds."""
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.get_embedding.return_value = [0.1] * 768

        checker = HealthChecker(db_path=self.db_path, vector_client=mock_client)

        result = checker.check_vector_api()

        self.assertTrue(result.healthy)
        self.assertIn("768", result.message)

    def test_check_vector_api_circuit_breaker_open(self):
        """check_vector_api should be unhealthy when circuit breaker is open."""
        mock_client = MagicMock()
        mock_client.is_available.return_value = False

        checker = HealthChecker(db_path=self.db_path, vector_client=mock_client)

        result = checker.check_vector_api()

        self.assertFalse(result.healthy)
        self.assertIn("circuit breaker", result.message.lower())

    def test_check_vector_api_error(self):
        """check_vector_api should be unhealthy when API fails."""
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.get_embedding.side_effect = Exception("Connection refused")

        checker = HealthChecker(db_path=self.db_path, vector_client=mock_client)

        result = checker.check_vector_api()

        self.assertFalse(result.healthy)
        self.assertIn("Connection refused", result.message)

    def test_check_chat_api_not_configured(self):
        """check_chat_api should be healthy when not configured."""
        checker = HealthChecker(db_path=self.db_path, chat_client=None)

        result = checker.check_chat_api()

        self.assertTrue(result.healthy)
        self.assertIn("Not configured", result.message)

    def test_check_chat_api_healthy(self):
        """check_chat_api should be healthy when API responds."""
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.complete.return_value = "OK"

        checker = HealthChecker(db_path=self.db_path, chat_client=mock_client)

        result = checker.check_chat_api()

        self.assertTrue(result.healthy)
        self.assertIn("Connected", result.message)

    def test_check_chat_api_circuit_breaker_open(self):
        """check_chat_api should be unhealthy when circuit breaker is open."""
        mock_client = MagicMock()
        mock_client.is_available.return_value = False

        checker = HealthChecker(db_path=self.db_path, chat_client=mock_client)

        result = checker.check_chat_api()

        self.assertFalse(result.healthy)
        self.assertIn("circuit breaker", result.message.lower())

    def test_check_configuration_healthy(self):
        """check_configuration should be healthy with valid config."""
        checker = HealthChecker(db_path=self.db_path)

        # Mock validate_config to return valid result
        with patch('rag_system.config.validate_config') as mock_validate:
            mock_validate.return_value = (True, [])
            result = checker.check_configuration()

        self.assertTrue(result.healthy)

    def test_check_all_returns_overall_status(self):
        """check_all should return overall health status."""
        checker = HealthChecker(db_path=self.db_path)

        result = checker.check_all()

        self.assertIn('healthy', result)
        self.assertIn('status', result)
        self.assertIn('checks', result)
        self.assertIsInstance(result['checks'], list)

    def test_check_all_healthy_when_all_pass(self):
        """check_all should be healthy when all checks pass."""
        checker = HealthChecker(db_path=self.db_path)

        # Mock validate_config to ensure configuration check passes
        with patch('rag_system.config.validate_config') as mock_validate:
            mock_validate.return_value = (True, [])
            result = checker.check_all()

        self.assertTrue(result['healthy'])
        self.assertEqual(result['status'], 'healthy')

    def test_check_all_unhealthy_when_any_fails(self):
        """check_all should be unhealthy when any check fails."""
        # Use invalid database path
        checker = HealthChecker(db_path="/")

        result = checker.check_all()

        self.assertFalse(result['healthy'])
        self.assertEqual(result['status'], 'unhealthy')


class TestFormatHealthReport(unittest.TestCase):
    """Tests for format_health_report function."""

    def test_format_healthy_report(self):
        """format_health_report should format healthy status."""
        result = {
            'healthy': True,
            'status': 'healthy',
            'checks': [
                {'name': 'database', 'healthy': True, 'message': 'Connected', 'details': {}},
                {'name': 'vector_api', 'healthy': True, 'message': 'Not configured', 'details': {}},
            ]
        }

        output = format_health_report(result)

        self.assertIn("HEALTHY", output)
        self.assertIn("[OK]", output)
        self.assertIn("Database", output)

    def test_format_unhealthy_report(self):
        """format_health_report should format unhealthy status."""
        result = {
            'healthy': False,
            'status': 'unhealthy',
            'checks': [
                {'name': 'database', 'healthy': False, 'message': 'Connection failed',
                 'details': {'errors': ['error1', 'error2']}},
            ]
        }

        output = format_health_report(result)

        self.assertIn("UNHEALTHY", output)
        self.assertIn("[FAIL]", output)


class TestCheckAPIAvailability(unittest.TestCase):
    """Tests for check_api_availability function."""

    def test_returns_true_when_not_required(self):
        """check_api_availability should return True when no APIs required."""
        result = check_api_availability(
            require_vector=False,
            require_chat=False
        )

        self.assertTrue(result)

    def test_returns_false_when_required_not_configured(self):
        """check_api_availability should return False when required API not configured."""
        result = check_api_availability(
            vector_client=None,
            require_vector=True
        )

        self.assertFalse(result)

    def test_returns_true_when_required_available(self):
        """check_api_availability should return True when required API available."""
        mock_client = MagicMock()
        mock_client.is_available.return_value = True

        result = check_api_availability(
            vector_client=mock_client,
            require_vector=True
        )

        self.assertTrue(result)

    def test_returns_false_when_circuit_breaker_open(self):
        """check_api_availability should return False when circuit breaker open."""
        mock_client = MagicMock()
        mock_client.is_available.return_value = False

        result = check_api_availability(
            vector_client=mock_client,
            require_vector=True
        )

        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
