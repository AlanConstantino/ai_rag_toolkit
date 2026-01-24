#!/bin/bash
# Run the test suite
#
# Usage:
#   ./scripts/run_tests.sh          # Run all tests
#   ./scripts/run_tests.sh -v       # Verbose output
#   ./scripts/run_tests.sh module   # Run specific test module

set -e
cd "$(dirname "$0")/.."

if [[ "$1" == "-v" ]]; then
    python -m unittest discover -s tests -p "test_*.py" -v
elif [[ -n "$1" ]]; then
    python -m unittest "tests.test_$1"
else
    python -m unittest discover -s tests -p "test_*.py"
fi
