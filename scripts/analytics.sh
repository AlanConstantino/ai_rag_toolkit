#!/bin/bash
# ==============================================================================
# analytics.sh - View query analytics and usage patterns
# ==============================================================================
#
# WHAT THIS DOES:
#   Shows analytics about how the RAG system has been used, including:
#   - Total number of queries
#   - Query types breakdown (factual, how-to, navigational, etc.)
#   - Average confidence scores
#   - Timing metrics (embedding, search, generation times)
#
# USAGE:
#   ./scripts/analytics.sh                      # Show analytics summary
#   ./scripts/analytics.sh --json               # Output as JSON
#   ./scripts/analytics.sh --export data.csv    # Export query logs to CSV
#
# OPTIONS:
#   --json              Output analytics as JSON (for scripting)
#   --export FILE       Export all query logs to a CSV file
#   --limit N           Limit export to N most recent queries
#
# EXAMPLE OUTPUT:
#   Query Analytics
#   ========================================
#   Total queries: 42
#   Queries with answers: 38
#   Average confidence: 85%
#
#   Query Types:
#     factual: 20
#     howto: 15
#     navigational: 7
#
#   Timing (average ms):
#     Total: 1250ms
#     Embedding: 150ms
#     Search: 50ms
#     Generation: 1000ms
#
# REQUIREMENTS:
#   - Python 3.6.5+
#   - Some queries must have been run first
#
# ==============================================================================

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

python -m rag_system.main analytics "$@"
