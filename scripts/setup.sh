#!/bin/bash
# ==============================================================================
# setup.sh - First-time setup wizard for the RAG system
# ==============================================================================
#
# WHAT THIS DOES:
#   Guides you through the initial setup of the RAG system:
#   1. Checks Python version
#   2. Helps you create a .env file with your API key
#   3. Initializes the database
#   4. Runs a health check
#
# USAGE:
#   ./scripts/setup.sh
#
# REQUIREMENTS:
#   - Python 3.6.5 or higher
#   - An OpenAI API key (get one at https://platform.openai.com/api-keys)
#
# AFTER SETUP:
#   1. Crawl some docs: ./scripts/crawl.sh https://docs.example.com
#   2. Ask questions:   ./scripts/query.sh "How do I get started?"
#   3. Or interactive:  ./scripts/interactive.sh
#
# ==============================================================================

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

echo "========================================"
echo "  RAG System - Setup Wizard"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 6 ]]; then
    echo "ERROR: Python 3.6.5+ is required (found $PYTHON_VERSION)"
    exit 1
fi
echo "  Python $PYTHON_VERSION"
echo ""

# Check for .env file
if [[ -f ".env" ]]; then
    echo "Found existing .env file"
    if grep -q "OPENAI_API_KEY" .env; then
        echo "  OPENAI_API_KEY is configured"
    else
        echo "  WARNING: OPENAI_API_KEY not found in .env"
    fi
else
    echo "No .env file found. Let's create one!"
    echo ""
    echo "You'll need an OpenAI API key for embeddings and answer generation."
    echo "Get one at: https://platform.openai.com/api-keys"
    echo ""
    read -p "Enter your OpenAI API key (or press Enter to skip): " API_KEY

    if [[ -n "$API_KEY" ]]; then
        echo "OPENAI_API_KEY=$API_KEY" > .env
        echo ""
        echo "Created .env file with your API key"
    else
        echo ""
        echo "Skipped API key setup. You can add it later:"
        echo "  echo 'OPENAI_API_KEY=sk-your-key' > .env"
    fi
fi
echo ""

# Initialize database
echo "Initializing database..."
python3 -c "from rag_system.database import init_db; init_db('rag_system.db')"
echo "  Database ready: rag_system.db"
echo ""

# Run health check
echo "Running health check..."
python -m rag_system.main health || true
echo ""

echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Crawl documentation:"
echo "     ./scripts/crawl.sh https://docs.python.org/3/"
echo ""
echo "  2. Ask questions:"
echo "     ./scripts/query.sh 'How do I get started?'"
echo ""
echo "  3. Or use interactive mode:"
echo "     ./scripts/interactive.sh"
echo ""
echo "For more help, see the README or run:"
echo "  python -m rag_system.main --help"
