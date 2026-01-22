"""Graceful shutdown handling for the RAG system.

Provides signal handlers and resource cleanup utilities.
"""

import atexit
import signal
import sys
from typing import Callable, List, Optional
import threading

from rag_system.utils import get_logger

logger = get_logger(__name__)


class ShutdownManager:
    """Manages graceful shutdown and resource cleanup."""

    def __init__(self):
        """Initialize the shutdown manager."""
        self._shutdown_requested = threading.Event()
        self._cleanup_handlers: List[Callable[[], None]] = []
        self._original_sigint = None
        self._original_sigterm = None
        self._installed = False
        self._shutdown_in_progress = False
        self._lock = threading.Lock()

    def install_handlers(self) -> None:
        """Install signal handlers for graceful shutdown.

        Handles SIGINT (Ctrl+C) and SIGTERM.
        """
        if self._installed:
            return

        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)
        self._installed = True
        logger.debug("Shutdown handlers installed")

    def uninstall_handlers(self) -> None:
        """Uninstall signal handlers and restore originals."""
        if not self._installed:
            return

        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

        try:
            atexit.unregister(self._cleanup)
        except Exception:
            pass

        self._installed = False
        logger.debug("Shutdown handlers uninstalled")

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        self._shutdown_requested.set()
        self._cleanup()

        # Re-raise signal for proper exit code
        if signum == signal.SIGINT:
            # For SIGINT, raise KeyboardInterrupt so callers can handle it
            raise KeyboardInterrupt()
        else:
            sys.exit(128 + signum)

    def register_cleanup(self, handler: Callable[[], None]) -> None:
        """Register a cleanup handler to be called on shutdown.

        Args:
            handler: Callable that performs cleanup.
        """
        with self._lock:
            if handler not in self._cleanup_handlers:
                self._cleanup_handlers.append(handler)

    def unregister_cleanup(self, handler: Callable[[], None]) -> None:
        """Unregister a cleanup handler.

        Args:
            handler: Handler to remove.
        """
        with self._lock:
            if handler in self._cleanup_handlers:
                self._cleanup_handlers.remove(handler)

    def _cleanup(self) -> None:
        """Execute all registered cleanup handlers."""
        with self._lock:
            if self._shutdown_in_progress:
                return
            self._shutdown_in_progress = True

        logger.info("Running cleanup handlers...")

        # Execute cleanup handlers in reverse order (LIFO)
        for handler in reversed(self._cleanup_handlers):
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in cleanup handler: {e}")

        logger.info("Cleanup complete")

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested.

        Returns:
            True if shutdown was requested via signal.
        """
        return self._shutdown_requested.is_set()

    def check_shutdown(self) -> None:
        """Check if shutdown was requested and raise if so.

        Raises:
            KeyboardInterrupt: If shutdown was requested.
        """
        if self._shutdown_requested.is_set():
            raise KeyboardInterrupt("Shutdown requested")


# Global shutdown manager instance
_shutdown_manager: Optional[ShutdownManager] = None


def get_shutdown_manager() -> ShutdownManager:
    """Get the global shutdown manager instance.

    Returns:
        ShutdownManager instance.
    """
    global _shutdown_manager
    if _shutdown_manager is None:
        _shutdown_manager = ShutdownManager()
    return _shutdown_manager


def install_shutdown_handlers() -> None:
    """Install signal handlers for graceful shutdown."""
    get_shutdown_manager().install_handlers()


def register_cleanup(handler: Callable[[], None]) -> None:
    """Register a cleanup handler to be called on shutdown.

    Args:
        handler: Callable that performs cleanup.
    """
    get_shutdown_manager().register_cleanup(handler)


def unregister_cleanup(handler: Callable[[], None]) -> None:
    """Unregister a cleanup handler.

    Args:
        handler: Handler to remove.
    """
    get_shutdown_manager().unregister_cleanup(handler)


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested.

    Returns:
        True if shutdown was requested.
    """
    return get_shutdown_manager().is_shutdown_requested()


def check_shutdown() -> None:
    """Check if shutdown was requested and raise if so.

    Raises:
        KeyboardInterrupt: If shutdown was requested.
    """
    get_shutdown_manager().check_shutdown()
