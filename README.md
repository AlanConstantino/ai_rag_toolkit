# Intelligent Documentation RAG System

A retrieval-augmented generation system for navigating and querying large documentation websites. Built with Python 3.6.5+ standard library only—no external dependencies.

## Features

- **Hybrid Search**: Combines BM25 lexical matching with vector similarity search
- **Knowledge Graph**: Extracts entities and relationships from documentation
- **Summarization Hierarchy**: Page, system, and global summaries for context
- **Query Intelligence**: Classifies, expands, and routes queries appropriately
- **Result Diversification**: Ensures answers draw from multiple sources

## Installation

No package installation required. Clone the repository and configure your API endpoints.

```bash
git clone <repository-url>
cd ai_rag_toolkit
```

## Configuration

Set environment variables for your vector and chat APIs:

```bash
export RAG_VECTOR_API_ENDPOINT="https://your-vector-api.com/embed"
export RAG_VECTOR_API_AUTH_VALUE="Bearer your-token"
export RAG_CHAT_API_ENDPOINT="https://your-chat-api.com/complete"
export RAG_CHAT_API_AUTH_VALUE="Bearer your-token"
```

Optional settings:
- `RAG_DATABASE_PATH` - SQLite database location (default: `rag_system.db`)
- `RAG_MAX_PAGES` - Maximum pages to crawl (default: 1000)
- `RAG_VECTOR_WEIGHT` / `RAG_BM25_WEIGHT` - Search weights (default: 0.7/0.3)

## Usage

### Ingest Documentation

Crawl and index a documentation website:

```bash
python -m rag_system.main ingest https://docs.example.com --max-pages 500
```

### Query the System

```bash
python -m rag_system.main query "How do I configure timeout settings?"
```

### Interactive Mode

```bash
python -m rag_system.main interactive
```

Commands in interactive mode:
- Type any question to query
- `stats` - Show database statistics
- `quit` or `exit` - Exit

### View Statistics

```bash
python -m rag_system.main stats
```

## Architecture

**Ingestion Pipeline:**
```
Website → Crawler → Parser → Chunker → Entity Extractor → Summarizer → Embedder → SQLite
```

**Query Pipeline:**
```
Query → Classifier → Expander → Hybrid Search → Diversifier → Reranker → Answer Generator
```

## Running Tests

```bash
# Run all tests
python -m unittest discover -s tests -p "test_*.py"

# Run specific test module
python -m unittest tests.test_database

# Run with verbose output
python -m unittest discover -s tests -p "test_*.py" -v
```

## Requirements

- Python 3.6.5+ (standard library only)
- Custom vector embedding API
- Custom chat/LLM completion API

## License

See LICENSE file for details.
