#!/bin/bash
# Run health checks
#
# Usage:
#   ./scripts/health.sh

set -e
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

python -m rag_system.main health
