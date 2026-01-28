#!/bin/bash
# First-time setup
#
# Usage:
#   ./scripts/setup.sh

set -e
cd "$(dirname "$0")/.."

echo "Setting up AI RAG Toolkit..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

# Create .env from example if it doesn't exist
if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        cp .env.example .env
        echo "Created .env from .env.example"
        echo "Edit .env to add your OPENAI_API_KEY"
    else
        echo "OPENAI_API_KEY=" > .env
        echo "Created empty .env file"
        echo "Add your OPENAI_API_KEY to .env"
    fi
else
    echo ".env already exists"
fi

# Make scripts executable
chmod +x scripts/*.sh

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Add your OPENAI_API_KEY to .env"
echo "  2. Crawl some docs: ./scripts/crawl.sh https://docs.example.com"
echo "  3. Query: ./scripts/query.sh \"your question\""
