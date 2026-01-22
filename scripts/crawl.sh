#!/bin/bash
# ==============================================================================
# crawl.sh - Crawl and ingest documentation from a website
# ==============================================================================
#
# WHAT THIS DOES:
#   Crawls a documentation website, extracts text content, breaks it into
#   searchable chunks, generates embeddings (if API key is set), and stores
#   everything in the local SQLite database.
#
# USAGE:
#   ./scripts/crawl.sh <url>                    # Crawl with default settings (max 1000 pages)
#   ./scripts/crawl.sh <url> --max-pages 100    # Limit to 100 pages
#   ./scripts/crawl.sh <url> --unlimited        # Crawl ALL pages (no limit)
#   ./scripts/crawl.sh <url> --ignore-robots    # Ignore robots.txt (use responsibly!)
#
# EXAMPLES:
#   ./scripts/crawl.sh https://docs.python.org/3/
#   ./scripts/crawl.sh https://fastapi.tiangolo.com --max-pages 50
#   ./scripts/crawl.sh https://docs.python.org/3.6/ --unlimited --ignore-robots
#
# REQUIREMENTS:
#   - Python 3.6.5+
#   - OPENAI_API_KEY in .env file (for embeddings, optional but recommended)
#
# WHAT HAPPENS:
#   1. Respects robots.txt by default
#   2. Crawls pages with 1 second delay between requests
#   3. Parses HTML and extracts main content
#   4. Breaks content into semantic chunks
#   5. Generates embeddings via OpenAI (if API key set)
#   6. Builds BM25 search index
#   7. Stores everything in rag_system.db
#
# NOTE:
#   If you crawl without OPENAI_API_KEY set, you can generate embeddings
#   later using: ./scripts/backfill.sh
#
# ==============================================================================

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

# Check if URL is provided
if [[ -z "$1" ]]; then
    echo "ERROR: Please provide a URL to crawl"
    echo ""
    echo "Usage: ./scripts/crawl.sh <url> [options]"
    echo ""
    echo "Examples:"
    echo "  ./scripts/crawl.sh https://docs.python.org/3/"
    echo "  ./scripts/crawl.sh https://docs.example.com --max-pages 100"
    exit 1
fi

URL="$1"
shift

echo "========================================"
echo "  RAG System - Website Crawler"
echo "========================================"
echo ""
echo "Target URL: $URL"
echo "Options: $@"
echo ""

# Check for .env file
if [[ -f ".env" ]]; then
    echo "Found .env file - API keys will be loaded automatically"
else
    echo "WARNING: No .env file found"
    echo "Embeddings won't be generated without OPENAI_API_KEY"
    echo "You can run ./scripts/backfill.sh later to add them"
fi
echo ""

echo "Starting crawl..."
echo ""

python -m rag_system.main ingest "$URL" "$@"

echo ""
echo "========================================"
echo "  Crawl complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  - Query: ./scripts/query.sh 'your question'"
echo "  - Interactive: ./scripts/interactive.sh"
echo "  - Stats: ./scripts/stats.sh"
