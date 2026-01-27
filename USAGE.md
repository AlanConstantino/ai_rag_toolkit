# Usage Guide

## Quick Start (5 minutes)

```bash
# 1. Setup
./scripts/setup.sh

# 2. Add your OpenAI API key
echo 'OPENAI_API_KEY=sk-your-key' >> .env

# 3. Crawl documentation
./scripts/crawl.sh https://docs.python.org/3/ --max-pages 50

# 4. Ask questions
./scripts/query.sh "How do I install Python?"
```

## Available Commands

| Command | What it does |
|---------|-------------|
| `./scripts/crawl.sh <url>` | Download and index a documentation site |
| `./scripts/query.sh "question"` | Ask a question (needs AI enabled) |
| `./scripts/search.sh "keywords"` | Search by keywords (no AI needed) |
| `./scripts/stats.sh` | Show how much data is indexed |
| `./scripts/interactive.sh` | Chat mode for multiple questions |

## Common Tasks

### Crawl a website
```bash
./scripts/crawl.sh https://docs.example.com
./scripts/crawl.sh https://docs.example.com --max-pages 100
./scripts/crawl.sh https://docs.example.com --unlimited
```

### Crawl authenticated websites (HTTP Basic Auth)
```bash
# Using username and password
./scripts/crawl.sh https://private.example.com \
    --basic-auth-user myuser \
    --basic-auth-pass mypassword

# Using a pre-encoded Base64 token
./scripts/crawl.sh https://private.example.com \
    --basic-auth-token dXNlcm5hbWU6cGFzc3dvcmQ=
```

### Search without AI
```bash
./scripts/search.sh "configuration"
./scripts/search.sh "timeout settings" --top-k 10
```

### Ask questions with AI
```bash
./scripts/query.sh "How do I configure timeouts?"
./scripts/query.sh "What are the installation steps?"
```

## Configuration

Edit `.env` to customize:

```bash
# Required for AI features
OPENAI_API_KEY=sk-your-key

# Optional: disable AI (use search only)
RAG_AI_ENABLED=false

# Optional: change database location
RAG_DATABASE_PATH=my_docs.db

# Optional: HTTP Basic Auth for authenticated sites
RAG_BASIC_AUTH_ENABLED=true
RAG_BASIC_AUTH_USERNAME=myuser
RAG_BASIC_AUTH_PASSWORD=mypassword
# Or use a pre-encoded token (takes precedence over username/password):
# RAG_BASIC_AUTH_TOKEN=dXNlcm5hbWU6cGFzc3dvcmQ=
```

See `.env.example` for all options.

## Troubleshooting

**No results from search?**
```bash
./scripts/rebuild-index.sh
```

**Want to start fresh?**
```bash
rm rag_system.db
./scripts/crawl.sh <url>
```

**Check system health:**
```bash
./scripts/health.sh
```
