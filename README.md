# AI RAG Toolkit

A documentation search system that crawls websites and lets you ask questions about them.

**Works with or without AI** - use keyword search alone, or add OpenAI for smarter answers.

## Quick Start

```bash
# Setup
git clone https://github.com/AlanConstantino/ai_rag_toolkit.git
cd ai_rag_toolkit
./scripts/setup.sh

# Add your OpenAI key (optional but recommended)
echo 'OPENAI_API_KEY=sk-your-key' >> .env

# Crawl some docs
./scripts/crawl.sh https://docs.python.org/3/ --max-pages 50

# Ask questions
./scripts/query.sh "How do I install Python?"
```

## Features

- **Hybrid Search** - Combines keyword matching (BM25) with AI semantic search
- **Works Offline** - Disable AI and use pure keyword search
- **Resume Crawls** - Interrupt with Ctrl+C, resume later
- **No Dependencies** - Pure Python 3.6+ standard library

## Scripts

| Script | Purpose |
|--------|---------|
| `setup.sh` | First-time setup |
| `crawl.sh` | Crawl a website |
| `query.sh` | Ask questions (AI) |
| `search.sh` | Keyword search (no AI) |
| `stats.sh` | Show statistics |
| `interactive.sh` | Chat mode |
| `health.sh` | Check system status |
| `rebuild-index.sh` | Fix search index |
| `run_tests.sh` | Run tests |

## Configuration

Create `.env` from the example:
```bash
cp .env.example .env
```

Key settings:
```bash
OPENAI_API_KEY=sk-your-key     # For AI features
RAG_AI_ENABLED=true            # Set false for keyword-only mode
RAG_MAX_PAGES=1000             # Crawl limit
```

See `.env.example` for all options.

## Crawling Authenticated Sites

For websites that require HTTP Basic Auth:

```bash
# Using username and password
./scripts/crawl.sh https://private.example.com \
    --basic-auth-user myuser \
    --basic-auth-pass mypassword

# Or using a pre-encoded Base64 token
./scripts/crawl.sh https://private.example.com \
    --basic-auth-token dXNlcm5hbWU6cGFzc3dvcmQ=
```

You can also set credentials via environment variables:
```bash
RAG_BASIC_AUTH_ENABLED=true
RAG_BASIC_AUTH_USERNAME=myuser
RAG_BASIC_AUTH_PASSWORD=mypassword
```

## Usage Guide

See [USAGE.md](USAGE.md) for detailed instructions.

## Requirements

- Python 3.6.5+
- OpenAI API key (optional, for AI features)

## License

See LICENSE file.
