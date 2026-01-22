#!/bin/bash
# ==============================================================================
# query.sh - Ask a question about your indexed documentation
# ==============================================================================
#
# WHAT THIS DOES:
#   Searches your indexed documentation using hybrid search (semantic + keyword)
#   and generates an AI-powered answer with citations.
#
# USAGE:
#   ./scripts/query.sh "your question here"
#   ./scripts/query.sh "how do I configure timeouts?" --top-k 10
#
# EXAMPLES:
#   ./scripts/query.sh "How do I install Python?"
#   ./scripts/query.sh "What are the configuration options?"
#   ./scripts/query.sh "How does authentication work?" --top-k 3
#
# OPTIONS:
#   --top-k N    Number of source chunks to retrieve (default: 5)
#
# REQUIREMENTS:
#   - Python 3.6.5+
#   - OPENAI_API_KEY in .env file (for answer generation)
#   - Previously crawled documentation (run crawl.sh first)
#
# HOW IT WORKS:
#   1. Classifies your query (factual, how-to, navigational, etc.)
#   2. Expands query for better recall
#   3. Searches using hybrid approach:
#      - Vector search (semantic similarity) - 70% weight
#      - BM25 search (keyword matching) - 30% weight
#   4. Reranks and diversifies results
#   5. Generates answer with citations using GPT
#
# ==============================================================================

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

# Check if question is provided
if [[ -z "$1" ]]; then
    echo "ERROR: Please provide a question"
    echo ""
    echo "Usage: ./scripts/query.sh \"your question here\""
    echo ""
    echo "Examples:"
    echo "  ./scripts/query.sh \"How do I install Python?\""
    echo "  ./scripts/query.sh \"What configuration options are available?\""
    exit 1
fi

QUESTION="$1"
shift

python -m rag_system.main query "$QUESTION" "$@"
