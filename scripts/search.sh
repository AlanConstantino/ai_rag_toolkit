#!/bin/bash
# BM25 keyword search (no AI required)
#
# Usage:
#   ./scripts/search.sh "search terms" [options]
#
# Examples:
#   ./scripts/search.sh "configuration timeout"
#   ./scripts/search.sh "install python" --top-k 10
#   ./scripts/search.sh "api endpoint" --json

set -e
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

if [[ -z "$1" ]]; then
    echo "Usage: ./scripts/search.sh \"search terms\" [options]"
    echo ""
    echo "Options:"
    echo "  --top-k N    Number of results (default: 5)"
    echo "  --json       Output as JSON"
    exit 1
fi

python -m rag_system.main bm25 "$@"
