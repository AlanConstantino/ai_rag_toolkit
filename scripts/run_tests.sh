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

# Default to discover all tests
TEST_TARGET="${1:-discover -s tests -p 'test_*.py'}"

# Check if verbose flag is passed
if [[ "$1" == "-v" ]]; then
    TEST_TARGET="discover -s tests -p 'test_*.py' -v"
elif [[ -n "$1" && "$1" != "-v" ]]; then
    # Specific test file provided
    TEST_TARGET="$1"
    shift
fi

echo "========================================"
echo "  Running RAG System Tests"
echo "========================================"
echo ""

python -m unittest $TEST_TARGET "$@"

echo ""
echo "========================================"
echo "  All tests passed!"
echo "========================================"
