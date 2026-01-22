#!/bin/bash
# ==============================================================================
# backfill.sh - Generate embeddings for chunks that are missing them
# ==============================================================================
#
# WHAT THIS DOES:
#   If you crawled a website without OPENAI_API_KEY set, your chunks won't
#   have embeddings and vector search won't work. This script generates
#   embeddings for all chunks that are missing them, WITHOUT re-crawling.
#
# WHEN TO USE:
#   - You crawled docs before setting up your API key
#   - You want to upgrade from BM25-only to hybrid search
#   - Some embeddings failed during initial crawl
#
# USAGE:
#   ./scripts/backfill.sh                   # Default batch size (100)
#   ./scripts/backfill.sh --batch-size 50   # Smaller batches (slower but safer)
#
# REQUIREMENTS:
#   - Python 3.6.5+
#   - OPENAI_API_KEY in .env file (required!)
#   - Previously crawled documentation with chunks in database
#
# HOW IT WORKS:
#   1. Finds all chunks in the database without embeddings
#   2. Builds contextual text (page title + heading path + content)
#   3. Sends batches to OpenAI for embedding generation
#   4. Saves embeddings back to the database
#   5. Now hybrid search will work!
#
# COST NOTE:
#   Uses OpenAI's text-embedding-3-small model.
#   Typical cost: ~$0.02 per 1M tokens (~$0.001 for 1000 chunks)
#
# ==============================================================================

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

# Check for .env file
if [[ ! -f ".env" ]]; then
    echo "ERROR: No .env file found"
    echo "Please create a .env file with your OPENAI_API_KEY"
    echo ""
    echo "Example:"
    echo "  echo 'OPENAI_API_KEY=sk-your-key-here' > .env"
    exit 1
fi

# Check if OPENAI_API_KEY is in .env
if ! grep -q "OPENAI_API_KEY" .env; then
    echo "ERROR: OPENAI_API_KEY not found in .env file"
    echo "Please add your OpenAI API key to the .env file"
    exit 1
fi

echo "========================================"
echo "  RAG System - Embedding Backfill"
echo "========================================"
echo ""

python -m rag_system.main backfill "$@"

echo ""
echo "========================================"
echo "  Backfill complete!"
echo "========================================"
echo ""
echo "Hybrid search is now enabled."
echo "Try: ./scripts/query.sh 'your question'"
