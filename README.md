# Intelligent Documentation RAG System

A retrieval-augmented generation (RAG) system for navigating and querying large documentation websites. Built with **Python 3.6.5+ standard library only**—no external dependencies except HTTP APIs.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/AlanConstantino/ai_rag_toolkit.git
cd ai_rag_toolkit
./scripts/setup.sh

# 2. Add your OpenAI API key to .env
echo 'OPENAI_API_KEY=sk-your-key-here' > .env

# 3. Crawl some documentation
./scripts/crawl.sh https://docs.python.org/3/ --max-pages 100

# 4. Ask questions
./scripts/query.sh "How do I install Python?"
```

## Features

### What It Does Well

- **Hybrid Search**: Combines BM25 (keyword matching) with vector similarity (semantic search) for best-of-both-worlds retrieval
- **No Dependencies**: Pure Python 3.6+ standard library—runs anywhere without pip install
- **Incremental Updates**: Re-crawling detects changed pages via content hashing and only re-indexes what changed
- **Production Ready**: Circuit breakers, rate limiting, graceful shutdown, health checks, structured logging
- **Semantic Chunking**: Preserves document structure with heading paths (e.g., "Config > Timeouts > Read Timeout")
- **Query Intelligence**: Classifies queries (factual, how-to, navigational) and expands them for better recall
- **Result Diversification**: Ensures answers draw from multiple pages, not just one source
- **Caching**: LRU cache for query results with configurable TTL

### Current Limitations

- **No Crawl Resume**: If you interrupt a crawl, it starts over (see [Issue #50](https://github.com/AlanConstantino/ai_rag_toolkit/issues/50))
- **Hallucination Risk**: LLM may not always use retrieved context faithfully (see [Issue #49](https://github.com/AlanConstantino/ai_rag_toolkit/issues/49))
- **Entity Extraction Not Wired**: Knowledge graph extraction code exists but isn't called during ingestion (see [Issue #25](https://github.com/AlanConstantino/ai_rag_toolkit/issues/25))
- **Summarization Not Wired**: Page/system/global summarization exists but isn't called (see [Issues #26-27](https://github.com/AlanConstantino/ai_rag_toolkit/issues/26))
- **Single-threaded Crawling**: No parallel fetching—respects rate limits but slower on large sites

## Installation

No package installation required. Just clone and configure.

```bash
git clone https://github.com/AlanConstantino/ai_rag_toolkit.git
cd ai_rag_toolkit
```

### Requirements

- Python 3.6.5 or higher
- OpenAI API key (or custom embedding/chat API endpoints)

## Configuration

### Option 1: OpenAI (Recommended)

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-your-key-here
```

The system auto-loads `.env` and uses OpenAI for embeddings (`text-embedding-3-small`) and chat (`gpt-4o-mini`).

### Option 2: Custom API Endpoints

For self-hosted or alternative APIs:

```bash
export RAG_VECTOR_API_ENDPOINT="https://your-vector-api.com/embed"
export RAG_VECTOR_API_AUTH_VALUE="Bearer your-token"
export RAG_CHAT_API_ENDPOINT="https://your-chat-api.com/complete"
export RAG_CHAT_API_AUTH_VALUE="Bearer your-token"
```

### All Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_DATABASE_PATH` | `rag_system.db` | SQLite database location |
| `RAG_MAX_PAGES` | `1000` | Default max pages to crawl |
| `RAG_CRAWL_DELAY_SECONDS` | `1.0` | Delay between requests |
| `RAG_VECTOR_WEIGHT` | `0.7` | Weight for vector search |
| `RAG_BM25_WEIGHT` | `0.3` | Weight for BM25 search |
| `RAG_SMALL_CHUNK_SIZE` | `500` | Small chunk size (chars) |
| `RAG_LARGE_CHUNK_SIZE` | `2000` | Large chunk size (chars) |
| `RAG_QUERY_CACHE_ENABLED` | `true` | Enable query result caching |
| `RAG_QUERY_CACHE_TTL` | `3600` | Cache TTL in seconds |

## Usage

### Using the Scripts (Recommended)

Convenience scripts are in `scripts/`:

```bash
# First-time setup
./scripts/setup.sh

# Crawl documentation
./scripts/crawl.sh https://docs.python.org/3/ --max-pages 100
./scripts/crawl.sh https://example.com --unlimited --ignore-robots

# Query
./scripts/query.sh "How do I configure timeouts?"

# Interactive mode
./scripts/interactive.sh

# View stats
./scripts/stats.sh

# Check health
./scripts/health.sh

# View analytics
./scripts/analytics.sh

# Run tests
./scripts/run_tests.sh
```

### Using Python Directly

```bash
# Ingest documentation
python -m rag_system.main ingest https://docs.example.com --max-pages 500

# Query
python -m rag_system.main query "How do I configure timeout settings?"

# Interactive mode
python -m rag_system.main interactive

# Show stats
python -m rag_system.main stats

# Health check
python -m rag_system.main health

# Query analytics
python -m rag_system.main analytics

# Backfill embeddings (if crawled without API key)
python -m rag_system.main backfill
```

### CLI Options

**Ingest command:**
```bash
python -m rag_system.main ingest <url> [options]

Options:
  --max-pages N      Maximum pages to crawl (default: 1000)
  --unlimited        Crawl all pages with no limit
  --ignore-robots    Ignore robots.txt restrictions
```

**Query command:**
```bash
python -m rag_system.main query "question" [options]

Options:
  --top-k N          Number of results to retrieve (default: 5)
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│  Website → Crawler → Parser → Chunker → Embedder → SQLite          │
│              │          │         │          │                      │
│         robots.txt   HTML→MD   semantic   OpenAI                    │
│         rate limit            + headings  vectors                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                              │
├─────────────────────────────────────────────────────────────────────┤
│  Query → Classifier → Expander → Hybrid Search → Reranker → LLM    │
│              │            │            │             │              │
│         factual/      synonyms    BM25 + Vector   diversify        │
│         howto/etc     related       (70/30)       sources          │
└─────────────────────────────────────────────────────────────────────┘
```

### Search Strategy

The system uses **hybrid search** combining:

1. **BM25 (30% weight)**: Traditional keyword matching using TF-IDF variant
   - Great for exact terms, code snippets, specific names

2. **Vector Search (70% weight)**: Semantic similarity using embeddings
   - Great for conceptual queries, synonyms, paraphrasing

Results are merged, deduplicated, reranked, and diversified to ensure answers come from multiple sources.

### Chunking Strategy

Documents are split into:
- **Small chunks** (~500 chars): Used for precise retrieval
- **Large chunks** (~2000 chars): Parent chunks for expanded context

Each chunk preserves its **heading path** (e.g., "Installation > Requirements > Python Version") for better context.

## Database Schema

The system uses SQLite with these key tables:

- **`pages`**: Crawled documentation pages with content hash for change detection
- **`chunks`**: Text chunks with embeddings and heading paths
- **`doc_terms`**: BM25 term frequency index
- **`entities`**: Extracted entities (not currently populated)
- **`relationships`**: Entity relationships (not currently populated)
- **`query_log`**: Query history for analytics

## Testing

```bash
# Run all tests (612 tests)
./scripts/run_tests.sh

# Run with verbose output
./scripts/run_tests.sh -v

# Run specific test file
python -m unittest tests.test_database

# Run specific test
python -m unittest tests.test_chunker.TestChunker.test_basic_chunking
```

## Project Structure

```
ai_rag_toolkit/
├── rag_system/
│   ├── main.py              # CLI entry point
│   ├── config.py            # Configuration settings
│   ├── database.py          # SQLite schema and operations
│   ├── api_client.py        # OpenAI/custom API clients
│   ├── utils.py             # Logging, metrics, helpers
│   ├── security.py          # Input validation
│   ├── health.py            # Health checks
│   ├── shutdown.py          # Graceful shutdown
│   ├── ingestion/
│   │   ├── crawler.py       # Web crawler with retry logic
│   │   ├── parser.py        # HTML parsing
│   │   ├── chunker.py       # Semantic chunking
│   │   ├── indexer.py       # Orchestrates ingestion
│   │   └── html_to_markdown.py
│   ├── search/
│   │   ├── bm25_search.py   # BM25 lexical search
│   │   ├── vector_search.py # Vector similarity search
│   │   ├── hybrid_search.py # Combined search
│   │   ├── reranker.py      # Result reranking
│   │   └── diversifier.py   # Source diversification
│   ├── query/
│   │   ├── classifier.py    # Query type classification
│   │   ├── expander.py      # Query expansion
│   │   ├── confidence.py    # Confidence scoring
│   │   ├── context_builder.py
│   │   └── answer_generator.py
│   ├── knowledge_graph/     # Entity extraction (not wired)
│   └── summarization/       # Summarization (not wired)
├── scripts/                 # Convenience bash scripts
├── tests/                   # Test suite (612 tests)
└── .env                     # API keys (create this)
```

## Known Issues & Pitfalls

### Robots.txt Blocks Old Docs
Some sites (like docs.python.org) block crawling of old versions via robots.txt:
```bash
# This won't crawl (robots.txt blocks /3.6/)
./scripts/crawl.sh https://docs.python.org/3.6/

# Use --ignore-robots to bypass (use responsibly)
./scripts/crawl.sh https://docs.python.org/3.6/ --ignore-robots
```

### Embeddings Require API Key
If you crawl without `OPENAI_API_KEY` set, chunks won't have embeddings and vector search won't work. Fix with:
```bash
./scripts/backfill.sh
```

### Large Sites Take Time
With 1-second crawl delay (respecting servers), 1000 pages = ~17 minutes. Use `--max-pages` to limit during testing.

### Memory Usage
Vector search loads all embeddings into memory. For very large indexes (100k+ chunks), consider increasing available RAM.

## Future Work

See [open issues](https://github.com/AlanConstantino/ai_rag_toolkit/issues) for planned improvements:

- **[#50](https://github.com/AlanConstantino/ai_rag_toolkit/issues/50)**: Crawler resume capability for interrupted crawls
- **[#49](https://github.com/AlanConstantino/ai_rag_toolkit/issues/49)**: Grounding safeguards to prevent hallucination
- **[#25-28](https://github.com/AlanConstantino/ai_rag_toolkit/issues/25)**: Wire up entity extraction, summarization, and knowledge graph
- **[#30-33](https://github.com/AlanConstantino/ai_rag_toolkit/issues/30)**: Code organization and documentation improvements

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `./scripts/run_tests.sh`
5. Submit a pull request

## License

See LICENSE file for details.
