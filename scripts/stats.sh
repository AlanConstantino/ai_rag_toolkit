#!/bin/bash
# ==============================================================================
# stats.sh - Show database and system statistics
# ==============================================================================
#
# WHAT THIS DOES:
#   Displays statistics about your indexed documentation including:
#   - Number of pages crawled
#   - Number of text chunks
#   - Number of entities extracted
#   - Number of relationships in knowledge graph
#
# USAGE:
#   ./scripts/stats.sh
#
# EXAMPLE OUTPUT:
#   RAG System Statistics
#   ==============================
#   Pages: 10
#   Chunks: 170
#   Entities: 0
#   Relationships: 0
#
# REQUIREMENTS:
#   - Python 3.6.5+
#   - Previously crawled documentation
#
# TIP:
#   If "Chunks" is high but you're getting poor search results,
#   try running ./scripts/backfill.sh to generate embeddings.
#
# ==============================================================================

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

python -m rag_system.main stats
