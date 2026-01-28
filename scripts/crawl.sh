#!/bin/bash
# Crawl and index a documentation website
#
# Usage:
#   ./scripts/crawl.sh <url> [options]
#
# Examples:
#   ./scripts/crawl.sh https://docs.python.org/3/
#   ./scripts/crawl.sh https://example.com --max-pages 50
#   ./scripts/crawl.sh https://example.com --unlimited
#   ./scripts/crawl.sh https://example.com --fresh

set -e
cd "$(dirname "$0")/.."

# Load .env if it exists
if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

if [[ -z "$1" ]]; then
    echo "Usage: ./scripts/crawl.sh <url> [options]"
    echo ""
    echo "Options:"
    echo "  --max-pages N    Limit pages to crawl (default: 1000)"
    echo "  --unlimited      Crawl all pages"
    echo "  --fresh          Start fresh, ignore saved session"
    echo "  --ignore-robots  Ignore robots.txt"
    exit 1
fi

python -m rag_system.main ingest "$@"
