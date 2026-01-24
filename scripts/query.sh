#!/bin/bash
# Query the RAG system with AI-powered answers
#
# Usage:
#   ./scripts/query.sh "your question"
#
# Examples:
#   ./scripts/query.sh "How do I configure timeouts?"
#   ./scripts/query.sh "What is the installation process?"
#
# Note: Requires RAG_AI_ENABLED=true and OPENAI_API_KEY set

set -e
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

if [[ -z "$1" ]]; then
    echo "Usage: ./scripts/query.sh \"your question\""
    echo ""
    echo "Requires OPENAI_API_KEY and RAG_AI_ENABLED=true"
    exit 1
fi

python -m rag_system.main query "$@"
