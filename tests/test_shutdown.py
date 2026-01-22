"""Tests for the shutdown module."""

import unittest
from unittest.mock import patch, MagicMock
import signal
import threading

from rag_system.shutdown import (
    ShutdownManager,
    get_shutdown_manager,
    install_shutdown_handlers,
    register_cleanup,
    unregister_cleanup,
    is_shutdown_requested,
)


class TestShutdownManager(unittest.TestCase):
    """Tests for ShutdownManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = ShutdownManager()

    def tearDown(self):
        """Clean up test fixtures."""
        self.manager.uninstall_handlers()

    def test_initial_state(self):
        """ShutdownManager should start with shutdown not requested."""
        self.assertFalse(self.manager.is_shutdown_requested())

    def test_install_handlers(self):
        """install_handlers should set up signal handlers."""
        self.manager.install_handlers()

        # Should be marked as installed
        self.assertTrue(self.manager._installed)

    def test_install_handlers_idempotent(self):
        """install_handlers should be idempotent."""
        self.manager.install_handlers()
        self.manager.install_handlers()  # Should not raise

        self.assertTrue(self.manager._installed)

    def test_uninstall_handlers(self):
        """uninstall_handlers should restore original handlers."""
        self.manager.install_handlers()
        self.manager.uninstall_handlers()

        self.assertFalse(self.manager._installed)

    def test_register_cleanup(self):
        """register_cleanup should add handler."""
        handler = MagicMock()

        self.manager.register_cleanup(handler)

        self.assertIn(handler, self.manager._cleanup_handlers)

    def test_register_cleanup_duplicate(self):
        """register_cleanup should not add duplicate handlers."""
        handler = MagicMock()

        self.manager.register_cleanup(handler)
        self.manager.register_cleanup(handler)  # Duplicate

        # Should only have one instance
        self.assertEqual(self.manager._cleanup_handlers.count(handler), 1)

    def test_unregister_cleanup(self):
        """unregister_cleanup should remove handler."""
        handler = MagicMock()
        self.manager.register_cleanup(handler)

        self.manager.unregister_cleanup(handler)

        self.assertNotIn(handler, self.manager._cleanup_handlers)

    def test_cleanup_handlers_called(self):
        """Cleanup handlers should be called during cleanup."""
        handler1 = MagicMock()
        handler2 = MagicMock()
        self.manager.register_cleanup(handler1)
        self.manager.register_cleanup(handler2)

        self.manager._cleanup()

        handler1.assert_called_once()
        handler2.assert_called_once()

    def test_cleanup_handlers_called_in_reverse_order(self):
        """Cleanup handlers should be called in LIFO order."""
        call_order = []
        handler1 = MagicMock(side_effect=lambda: call_order.append(1))
        handler2 = MagicMock(side_effect=lambda: call_order.append(2))
        self.manager.register_cleanup(handler1)
        self.manager.register_cleanup(handler2)

        self.manager._cleanup()

        # LIFO order: handler2 first, then handler1
        self.assertEqual(call_order, [2, 1])

    def test_cleanup_continues_on_error(self):
        """Cleanup should continue even if a handler raises."""
        handler1 = MagicMock(side_effect=Exception("Error in handler 1"))
        handler2 = MagicMock()
        self.manager.register_cleanup(handler1)
        self.manager.register_cleanup(handler2)

        # Should not raise
        self.manager._cleanup()

        # Both handlers should have been called
        handler1.assert_called_once()
        handler2.assert_called_once()

    def test_cleanup_only_runs_once(self):
        """Cleanup should only run once."""
        handler = MagicMock()
        self.manager.register_cleanup(handler)

        self.manager._cleanup()
        self.manager._cleanup()  # Second call

        # Handler should only be called once
        handler.assert_called_once()

    def test_check_shutdown_does_not_raise_normally(self):
        """check_shutdown should not raise when shutdown not requested."""
        # Should not raise
        self.manager.check_shutdown()

    def test_check_shutdown_raises_when_requested(self):
        """check_shutdown should raise KeyboardInterrupt when shutdown requested."""
        self.manager._shutdown_requested.set()

        with self.assertRaises(KeyboardInterrupt):
            self.manager.check_shutdown()


class TestModuleFunctions(unittest.TestCase):
    """Tests for module-level functions."""

    def test_get_shutdown_manager_returns_singleton(self):
        """get_shutdown_manager should return the same instance."""
        manager1 = get_shutdown_manager()
        manager2 = get_shutdown_manager()

        self.assertIs(manager1, manager2)

    def test_is_shutdown_requested_returns_bool(self):
        """is_shutdown_requested should return a boolean."""
        result = is_shutdown_requested()

        self.assertIsInstance(result, bool)


class TestManagedConnection(unittest.TestCase):
    """Tests for managed_connection context manager."""

    def test_connection_auto_closed(self):
        """managed_connection should close connection on exit."""
        from rag_system.database import managed_connection
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from rag_system.database import init_db
            init_db(db_path)

            with managed_connection(db_path) as conn:
                # Connection should be open
                cursor = conn.execute("SELECT 1")
                self.assertEqual(cursor.fetchone()[0], 1)

            # Connection should be closed now - verify by trying to use it
            # This should raise because connection is closed
            with self.assertRaises(Exception):
                conn.execute("SELECT 1")
        finally:
            os.unlink(db_path)

    def test_connection_closed_on_exception(self):
        """managed_connection should close connection even on exception."""
        from rag_system.database import managed_connection
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from rag_system.database import init_db
            init_db(db_path)

            conn_ref = None
            try:
                with managed_connection(db_path) as conn:
                    conn_ref = conn
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Connection should be closed even after exception
            with self.assertRaises(Exception):
                conn_ref.execute("SELECT 1")
        finally:
            os.unlink(db_path)


if __name__ == '__main__':
    unittest.main()
