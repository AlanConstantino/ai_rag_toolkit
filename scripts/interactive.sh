#!/bin/bash
# Start interactive query mode
#
# Usage:
#   ./scripts/interactive.sh

set -e
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

python -m rag_system.main interactive
