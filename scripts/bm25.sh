#!/bin/bash
# ==============================================================================
# bm25.sh - Perform BM25 lexical search (no AI required)
# ==============================================================================
#
# WHAT THIS DOES:
#   Searches your indexed documentation using BM25 (lexical/keyword matching).
#   This is pure text matching - no embeddings, no LLM, no API calls.
#   Works offline and is very fast.
#
# USAGE:
#   ./scripts/bm25.sh "your search query"
#   ./scripts/bm25.sh "python install" --top-k 10
#   ./scripts/bm25.sh "configuration" --json
#
# OPTIONS:
#   --top-k N    Number of results to return (default: 5)
#   --json       Output results as JSON
#
# EXAMPLES:
#   ./scripts/bm25.sh "how to install"
#   ./scripts/bm25.sh "timeout configuration" --top-k 3
#   ./scripts/bm25.sh "authentication" --json
#
# REQUIREMENTS:
#   - Python 3.6.5+
#   - Previously crawled documentation (run crawl.sh first)
#   - NO API key required!
#
# HOW IT WORKS:
#   BM25 (Best Matching 25) is a ranking algorithm that scores documents
#   based on term frequency and inverse document frequency. It finds
#   documents containing your search terms, with higher scores for:
#   - Rare terms (more distinctive)
#   - Multiple occurrences of the term
#   - Shorter documents (more focused content)
#
# ==============================================================================

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

# Check if query is provided
if [[ -z "$1" ]]; then
    echo "ERROR: Please provide a search query"
    echo ""
    echo "Usage: ./scripts/bm25.sh \"your search query\""
    echo ""
    echo "Examples:"
    echo "  ./scripts/bm25.sh \"how to install\""
    echo "  ./scripts/bm25.sh \"configuration\" --top-k 10"
    exit 1
fi

QUERY="$1"
shift

python -m rag_system.main bm25 "$QUERY" "$@"
