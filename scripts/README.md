# RAG System Scripts

Convenience scripts for interacting with the RAG (Retrieval-Augmented Generation) system.

## Quick Start

```bash
# First time? Run setup
./scripts/setup.sh

# Crawl some documentation
./scripts/crawl.sh https://docs.python.org/3/ --max-pages 50

# Ask questions
./scripts/query.sh "How do I install Python?"

# Or use interactive mode
./scripts/interactive.sh
```

## Available Scripts

| Script | Purpose |
|--------|---------|
| `setup.sh` | First-time setup wizard (creates .env, initializes DB) |
| `crawl.sh` | Crawl and index a documentation website |
| `query.sh` | Ask a single question (uses AI if available) |
| `bm25.sh` | BM25 lexical search (no AI required) |
| `interactive.sh` | Start an interactive Q&A session |
| `backfill.sh` | Generate embeddings for chunks missing them |
| `stats.sh` | Show database statistics |
| `health.sh` | Check system health and configuration |
| `analytics.sh` | View query analytics and usage patterns |
| `run_tests.sh` | Run the test suite |

## Script Details

### setup.sh
Run this first! Checks your Python version, helps create a `.env` file with your OpenAI API key, and initializes the database.

### crawl.sh
```bash
./scripts/crawl.sh <url> [options]

# Examples
./scripts/crawl.sh https://docs.python.org/3/
./scripts/crawl.sh https://fastapi.tiangolo.com --max-pages 100
./scripts/crawl.sh https://example.com --ignore-robots  # Use responsibly!
./scripts/crawl.sh https://example.com --fresh          # Start fresh crawl
```

**Resume support:** If you interrupt a crawl with Ctrl+C, progress is saved automatically. Run the same command again to resume. Use `--fresh` to start over.

### query.sh
```bash
./scripts/query.sh "your question here"
./scripts/query.sh "How does authentication work?" --top-k 10
```

### bm25.sh
Pure BM25 lexical search - no AI, no API keys required. Fast and works offline.

```bash
./scripts/bm25.sh "search terms"
./scripts/bm25.sh "configuration timeout" --top-k 10
./scripts/bm25.sh "authentication" --json
```

If you get no results, you may need to rebuild the index:
```bash
python -m rag_system.main rebuild-index
```

### backfill.sh
If you crawled before setting `OPENAI_API_KEY`, embeddings weren't generated. This script adds them without re-crawling.

```bash
./scripts/backfill.sh                   # Default batch size
./scripts/backfill.sh --batch-size 50   # Smaller batches
```

### analytics.sh
```bash
./scripts/analytics.sh                  # Summary view
./scripts/analytics.sh --json           # JSON output
./scripts/analytics.sh --export logs.csv  # Export to CSV
```

## Requirements

- Python 3.6.5+
- OpenAI API key (for embeddings and answer generation)

## Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-your-key-here
```

The scripts automatically load this file.
