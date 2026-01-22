#!/bin/bash
# ==============================================================================
# interactive.sh - Start an interactive Q&A session
# ==============================================================================
#
# WHAT THIS DOES:
#   Launches an interactive REPL where you can ask multiple questions
#   without restarting the system each time. Great for exploring your
#   indexed documentation.
#
# USAGE:
#   ./scripts/interactive.sh
#
# INTERACTIVE COMMANDS:
#   <any text>     Ask a question (just type and press Enter)
#   stats          Show database statistics
#   cache          Show query cache statistics
#   cache clear    Clear the query cache
#   help           Show available commands
#   quit / exit    Exit the session
#
# REQUIREMENTS:
#   - Python 3.6.5+
#   - OPENAI_API_KEY in .env file
#   - Previously crawled documentation
#
# TIP:
#   The system caches query results, so repeated questions are instant!
#
# ==============================================================================

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

echo "========================================"
echo "  RAG System - Interactive Mode"
echo "========================================"
echo ""
echo "Type your questions and press Enter."
echo "Type 'help' for commands, 'quit' to exit."
echo ""

python -m rag_system.main interactive
