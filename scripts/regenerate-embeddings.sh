#!/bin/bash
# Regenerate all embeddings with contextual retrieval
#
# This script forces regeneration of embeddings for all chunks,
# even those that already have embeddings. Use this after upgrading
# to contextual retrieval to improve search quality.
#
# Usage:
#   ./scripts/regenerate-embeddings.sh [options]
#
# Options:
#   --batch-size N    Number of chunks to embed per API call (default: 100)
#
# Examples:
#   ./scripts/regenerate-embeddings.sh
#   ./scripts/regenerate-embeddings.sh --batch-size 50

set -e
cd "$(dirname "$0")/.."

# Load .env if it exists
if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "Regenerating all embeddings with contextual retrieval..."
echo "This will re-embed all chunks, which may take a while and use API credits."
echo ""

python -m rag_system.main backfill --force "$@"
