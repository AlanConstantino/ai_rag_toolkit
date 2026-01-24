#!/bin/bash
# Rebuild the BM25 search index
#
# Usage:
#   ./scripts/rebuild-index.sh

set -e
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

python -m rag_system.main rebuild-index
