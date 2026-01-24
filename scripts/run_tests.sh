#!/bin/bash
# ==============================================================================
# run_tests.sh - Run the test suite for the RAG system
# ==============================================================================
#
# WHAT THIS DOES:
#   Runs all unit, integration, and end-to-end tests for the RAG system.
#   Use this to verify everything is working correctly after making changes.
#
# USAGE:
#   ./scripts/run_tests.sh              # Run all tests
#   ./scripts/run_tests.sh -v           # Run with verbose output
#   ./scripts/run_tests.sh tests/test_database.py  # Run specific test file
#
# REQUIREMENTS:
#   - Python 3.6.5+
#   - No external dependencies (standard library only)
#
# EXIT CODES:
#   0 = All tests passed
#   1 = Some tests failed
#
# ==============================================================================

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

echo "========================================"
echo "  Running RAG System Tests"
echo "========================================"
echo ""

# Handle arguments
if [[ -z "$1" ]]; then
    # No arguments - run all tests
    python -m unittest discover -s tests -p "test_*.py"
elif [[ "$1" == "-v" ]]; then
    # Verbose flag - run all tests with verbose
    python -m unittest discover -s tests -p "test_*.py" -v
else
    # Specific test file or pattern provided
    python -m unittest "$@"
fi

echo ""
echo "========================================"
echo "  All tests passed!"
echo "========================================"
