#!/bin/bash
# Show database statistics
#
# Usage:
#   ./scripts/stats.sh

set -e
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

python -m rag_system.main stats
