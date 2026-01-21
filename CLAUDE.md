# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# Intelligent Documentation RAG System

A retrieval-augmented generation system for navigating and querying large documentation websites. Built with Python 3.6.5 standard library only, using custom vector and chat APIs.

## Development Approach

**Test-Driven Development (TDD):** All modules must be developed using TDD:
1. Write failing tests first
2. Implement minimum code to pass tests
3. Refactor while keeping tests green

**No External Dependencies:** Use Python 3.6.5+ standard library only. All external functionality (embeddings, LLM completions) comes through custom HTTP APIs.

**Running Tests:**
```bash
# Run all tests
python -m unittest discover -s tests -p "test_*.py"

# Run specific test module
python -m unittest tests.test_database

# Run specific test class
python -m unittest tests.test_chunker.TestChunker

# Run with verbose output
python -m unittest discover -s tests -p "test_*.py" -v
```

**Running the Application:**
```bash
# Crawl a documentation site
python -m rag_system.main crawl https://docs.example.com --max-pages 500

# Build the full index
python -m rag_system.main index

# Query the system
python -m rag_system.main query "How do I configure timeout settings?"

# Interactive mode
python -m rag_system.main interactive

# Show database statistics
python -m rag_system.main status
```

## Overview

This system solves two problems:
1. **Navigation** - Hard to find information across hundreds of documentation pages
2. **Comprehension** - Technical documentation is often difficult to understand

Instead of basic keyword search, this system:
- Builds a **knowledge graph** of entities and relationships
- Creates a **summarization hierarchy** (page → system → global)
- Uses **hybrid search** (BM25 + vector similarity)
- Expands queries intelligently before searching
- Generates natural language answers with citations

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           INGESTION PHASE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Website] → [Crawler] → [Parser] → [Chunker]                      │
│                              ↓                                      │
│                    [Entity Extractor] ←── Chat API                 │
│                              ↓                                      │
│                    [Summarizer] ←── Chat API                       │
│                              ↓                                      │
│                    [Embedder] ←── Vector API                       │
│                              ↓                                      │
│                    [SQLite Database]                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                            QUERY PHASE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [User Query]                                                       │
│       ↓                                                             │
│  [Query Classifier] ←── Chat API                                   │
│       ↓                                                             │
│  [Query Expander] ←── Chat API                                     │
│       ↓                                                             │
│  [Hybrid Search]                                                    │
│       ├── Vector Search (cosine similarity)                        │
│       └── BM25 Search (lexical matching)                           │
│       ↓                                                             │
│  [Result Merger + Diversifier + Reranker]                          │
│       ↓                                                             │
│  [Confidence Check] ←── Chat API                                   │
│       ↓                                                             │
│  [Context Assembler]                                               │
│       ├── Global summary                                           │
│       ├── Relevant system summaries                                │
│       ├── Retrieved chunks (with heading paths)                    │
│       └── Related entities from knowledge graph                    │
│       ↓                                                             │
│  [Answer Generator] ←── Chat API                                   │
│       ↓                                                             │
│  [Response + Citations]                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Knowledge Graph
Extracts entities (systems, configurations, concepts) and relationships from documentation. Enables queries like "what depends on X?" and provides richer context for answers.

### 2. Summarization Hierarchy
- **Page summaries** - 2-3 sentence summary of each page
- **System summaries** - Summary of each logical system/component
- **Global summary** - Overview of the entire documentation

This allows the system to answer both specific and broad questions.

### 3. Hybrid Search (BM25 + Vector)
- **BM25** - Catches exact terminology matches ("TIMEOUT_CONFIG_V2")
- **Vector search** - Catches semantic similarity ("timeout settings")
- **Combined** - Best of both worlds

### 4. Smart Chunking
- Respects semantic boundaries (paragraphs, headings)
- Stores heading path with each chunk ("Configuration > Timeouts > Read Timeout")
- Parent-child chunks: small chunks for precise retrieval, large chunks for context

### 5. Query Intelligence
- **Classification** - Routes factual, how-to, global, and navigational queries differently
- **Expansion** - Rewrites queries multiple ways to improve recall
- **Confidence checking** - Detects when the answer isn't in the documentation

### 6. Result Diversification
Ensures results come from multiple pages, not just the top-matching single page.

---

## Project Structure

```
rag_system/
├── config.py                  # Configuration (API endpoints, SSL, etc.)
├── database.py                # SQLite setup and helpers
├── api_client.py              # Wrappers for vector and chat APIs
├── utils.py                   # Logging, JSON helpers, defensive parsing
│
├── ingestion/
│   ├── __init__.py
│   ├── crawler.py             # Web crawler (urllib, respects robots.txt)
│   ├── parser.py              # HTML to text extraction
│   ├── chunker.py             # Semantic chunking with heading paths
│   ├── entity_extractor.py    # LLM-based entity/relationship extraction
│   ├── summarizer.py          # Page, system, and global summarization
│   └── indexer.py             # Orchestrates full ingestion pipeline
│
├── search/
│   ├── __init__.py
│   ├── vector_search.py       # Cosine similarity search
│   ├── bm25_search.py         # BM25 lexical search
│   ├── hybrid_search.py       # Combines vector + BM25
│   ├── diversifier.py         # Ensures result diversity
│   └── reranker.py            # Final reranking logic
│
├── query/
│   ├── __init__.py
│   ├── classifier.py          # Query type classification
│   ├── expander.py            # LLM-based query expansion
│   ├── confidence.py          # Answer confidence checking
│   ├── context_builder.py     # Assembles context from multiple sources
│   └── answer_generator.py    # Generates final answer with citations
│
├── knowledge_graph/
│   ├── __init__.py
│   ├── graph_queries.py       # Traverse relationships
│   └── entity_resolver.py     # Basic entity deduplication
│
└── main.py                    # CLI entry point
```

---

## Database Schema

```sql
-- ============================================
-- CORE CONTENT
-- ============================================

CREATE TABLE pages (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    raw_html TEXT,
    parsed_text TEXT,
    summary TEXT,
    content_hash TEXT,              -- For incremental updates
    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    page_id INTEGER NOT NULL,
    parent_chunk_id INTEGER,        -- NULL for large chunks
    chunk_type TEXT,                -- 'large' or 'small'
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    heading_path TEXT,              -- e.g., "Configuration > Timeouts"
    embedding_json TEXT,            -- JSON array of floats
    FOREIGN KEY (page_id) REFERENCES pages(id),
    FOREIGN KEY (parent_chunk_id) REFERENCES chunks(id)
);

-- ============================================
-- KNOWLEDGE GRAPH
-- ============================================

CREATE TABLE entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT,               -- system, config, concept, process, tool
    description TEXT,
    page_id INTEGER,
    FOREIGN KEY (page_id) REFERENCES pages(id)
);

CREATE TABLE relationships (
    id INTEGER PRIMARY KEY,
    source_entity_id INTEGER NOT NULL,
    target_entity_id INTEGER NOT NULL,
    relationship_type TEXT,         -- depends_on, configures, part_of, etc.
    description TEXT,
    FOREIGN KEY (source_entity_id) REFERENCES entities(id),
    FOREIGN KEY (target_entity_id) REFERENCES entities(id)
);

CREATE TABLE chunk_entities (
    chunk_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, entity_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);

-- ============================================
-- SUMMARIZATION HIERARCHY
-- ============================================

CREATE TABLE systems (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    summary TEXT
);

CREATE TABLE page_systems (
    page_id INTEGER NOT NULL,
    system_id INTEGER NOT NULL,
    PRIMARY KEY (page_id, system_id),
    FOREIGN KEY (page_id) REFERENCES pages(id),
    FOREIGN KEY (system_id) REFERENCES systems(id)
);

CREATE TABLE global_summary (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- BM25 INDEX
-- ============================================

CREATE TABLE doc_terms (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL,
    term TEXT NOT NULL,
    term_frequency INTEGER NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);

CREATE TABLE corpus_stats (
    id INTEGER PRIMARY KEY,
    total_docs INTEGER,
    avg_doc_length REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE term_doc_frequencies (
    term TEXT PRIMARY KEY,
    doc_frequency INTEGER NOT NULL
);

-- ============================================
-- CACHING & LOGGING
-- ============================================

CREATE TABLE query_cache (
    query_hash TEXT PRIMARY KEY,
    query_type TEXT,                -- factual, howto, global, navigational
    expanded_queries TEXT,          -- JSON
    embedding_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE query_log (
    id INTEGER PRIMARY KEY,
    query TEXT,
    query_type TEXT,
    expanded_queries TEXT,
    retrieved_chunk_ids TEXT,       -- JSON array
    confidence_score REAL,
    answer_generated BOOLEAN,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- INDEXES
-- ============================================

CREATE INDEX idx_chunks_page ON chunks(page_id);
CREATE INDEX idx_chunks_parent ON chunks(parent_chunk_id);
CREATE INDEX idx_entities_name ON entities(name);
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_doc_terms_term ON doc_terms(term);
CREATE INDEX idx_doc_terms_chunk ON doc_terms(chunk_id);
```

---

## Core Algorithms

### BM25 Scoring

```python
import math

# Tunable parameters
K1 = 1.5  # Term frequency saturation (typical: 1.2-2.0)
B = 0.75  # Length normalization (typical: 0.75)

def bm25_score(query_terms, doc_term_freqs, doc_length, avg_doc_length, 
               doc_frequencies, total_docs):
    """
    Calculate BM25 score for a document given a query.
    
    Args:
        query_terms: list of query tokens
        doc_term_freqs: dict of {term: frequency} for this document
        doc_length: number of terms in this document
        avg_doc_length: average across corpus
        doc_frequencies: dict of {term: num_docs_containing_term}
        total_docs: total documents in corpus
    
    Returns:
        float: BM25 score
    """
    score = 0.0
    
    for term in query_terms:
        if term not in doc_term_freqs:
            continue
        
        tf = doc_term_freqs[term]
        df = doc_frequencies.get(term, 0)
        
        # IDF with smoothing
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
        
        # TF with saturation and length normalization
        numerator = tf * (K1 + 1)
        denominator = tf + K1 * (1 - B + B * (doc_length / avg_doc_length))
        tf_component = numerator / denominator
        
        score += idf * tf_component
    
    return score
```

### Cosine Similarity

```python
import math

def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)
```

### Hybrid Search Merging

```python
def hybrid_search(query, vector_results, bm25_results, 
                  vector_weight=0.7, bm25_weight=0.3):
    """
    Merge vector and BM25 results with score normalization.
    
    Args:
        query: original query string
        vector_results: list of (chunk_id, score) from vector search
        bm25_results: list of (chunk_id, score) from BM25 search
        vector_weight: weight for vector scores
        bm25_weight: weight for BM25 scores
    
    Returns:
        list of (chunk_id, combined_score) sorted descending
    """
    # Normalize scores to 0-1 range
    def normalize(results):
        if not results:
            return {}
        scores = [score for _, score in results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return {chunk_id: 1.0 for chunk_id, _ in results}
        return {
            chunk_id: (score - min_s) / (max_s - min_s) 
            for chunk_id, score in results
        }
    
    vector_normalized = normalize(vector_results)
    bm25_normalized = normalize(bm25_results)
    
    # Combine scores
    all_chunk_ids = set(vector_normalized.keys()) | set(bm25_normalized.keys())
    combined = {}
    
    for chunk_id in all_chunk_ids:
        v_score = vector_normalized.get(chunk_id, 0.0)
        b_score = bm25_normalized.get(chunk_id, 0.0)
        combined[chunk_id] = (v_score * vector_weight) + (b_score * bm25_weight)
    
    # Sort by combined score
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

### Result Diversification

```python
def diversify_results(ranked_results, chunk_to_page, max_per_page=2, top_k=10):
    """
    Ensure results are diverse across pages.
    
    Args:
        ranked_results: list of (chunk_id, score) sorted by score
        chunk_to_page: dict mapping chunk_id to page_id
        max_per_page: maximum chunks from any single page
        top_k: number of results to return
    
    Returns:
        list of (chunk_id, score)
    """
    diversified = []
    page_counts = {}
    
    for chunk_id, score in ranked_results:
        page_id = chunk_to_page.get(chunk_id)
        current_count = page_counts.get(page_id, 0)
        
        if current_count < max_per_page:
            diversified.append((chunk_id, score))
            page_counts[page_id] = current_count + 1
        
        if len(diversified) >= top_k:
            break
    
    return diversified
```

---

## LLM Prompts

### Entity Extraction

```
Analyze this documentation page and extract structured information.

TEXT:
{page_text}

Extract:
1. ENTITIES - Important nouns representing systems, services, configurations, 
   concepts, or tools mentioned in this documentation.
2. RELATIONSHIPS - How these entities connect to each other.

Format your response as JSON:
{
  "entities": [
    {
      "name": "exact name as it appears",
      "type": "system|config|concept|process|tool",
      "description": "one sentence description"
    }
  ],
  "relationships": [
    {
      "source": "entity name",
      "target": "entity name",
      "type": "depends_on|configures|part_of|connects_to|triggers|reads_from|writes_to",
      "description": "brief description of the relationship"
    }
  ]
}

Return ONLY valid JSON, no other text.
```

### Page Summarization

```
Summarize this documentation page in 2-3 concise sentences.

Focus on:
- What system or feature this page documents
- Key configuration options or settings mentioned
- Important behaviors, constraints, or warnings

TEXT:
{page_text}

SUMMARY:
```

### Query Classification

```
Classify this documentation search query into one of four types:

FACTUAL - Looking for a specific fact, setting, or value
  Examples: "What is the default timeout?", "What port does X use?"

HOWTO - Looking for steps or process explanation
  Examples: "How do I configure SSL?", "How to enable logging?"

GLOBAL - Asking about overall system, themes, or architecture
  Examples: "What are the main components?", "Give me an overview of X"

NAVIGATIONAL - Looking for where documentation exists
  Examples: "Where is the auth documentation?", "Find the API reference"

Query: "{query}"

Respond with ONLY one word: FACTUAL, HOWTO, GLOBAL, or NAVIGATIONAL
```

### Query Expansion

```
A user is searching technical documentation. Help improve their search.

Original query: "{query}"

Provide:
1. Three alternative phrasings that might match documentation better
2. Any specific system or component names that seem relevant
3. Related technical terms worth searching for

Format as JSON:
{
  "alternative_queries": ["...", "...", "..."],
  "detected_entities": ["..."],
  "related_terms": ["..."]
}

Return ONLY valid JSON.
```

### Confidence Check

```
You are evaluating whether retrieved documentation chunks can answer a query.

Query: "{query}"

Retrieved chunks:
---
{chunks}
---

Rate from 1-5 how confident you are the answer is in these chunks:
1 - Answer definitely not present
2 - Answer probably not present
3 - Uncertain
4 - Answer probably present
5 - Answer definitely present

Respond with ONLY a single number (1-5).
```

### Answer Generation

```
You are a documentation assistant. Answer the user's question using ONLY 
the provided context. Do not use any outside knowledge.

RULES:
- If the answer is not in the context, say "I couldn't find this information 
  in the documentation."
- Always cite which page(s) your answer comes from using [Source: page title] format
- Be concise but complete
- If information seems contradictory, note the discrepancy

CONTEXT:

Global Overview:
{global_summary}

Relevant System Information:
{system_summaries}

Related Entities:
{entities}

Documentation Excerpts:
{chunks}

---

QUESTION: {question}

ANSWER:
```

---

## CLI Usage

```bash
# Crawl a documentation site
python main.py crawl https://docs.example.com --max-pages 500

# Build the full index (chunks, embeddings, entities, summaries)
python main.py index

# Query the system
python main.py query "How do I configure the timeout settings?"

# Interactive mode
python main.py interactive

# Show database statistics
python main.py status

# Rebuild summaries (after adding new pages)
python main.py rebuild-summaries

# Export knowledge graph (for visualization)
python main.py export-graph --format json --output graph.json
```

---

## Configuration

Create `config.py` with your API details:

```python
# API Configuration
VECTOR_API_ENDPOINT = "https://your-vector-api.com/embed"
VECTOR_API_AUTH_HEADER = "Authorization"
VECTOR_API_AUTH_VALUE = "Bearer your-token"

CHAT_API_ENDPOINT = "https://your-chat-api.com/complete"
CHAT_API_AUTH_HEADER = "Authorization"
CHAT_API_AUTH_VALUE = "Bearer your-token"

# SSL Configuration (if needed)
SSL_CERT_PATH = "/path/to/cert.pem"  # or None
SSL_VERIFY = True  # Set False to disable verification (not recommended)

# Database
DATABASE_PATH = "rag_system.db"

# Crawler Settings
CRAWL_DELAY_SECONDS = 1.0
MAX_PAGES = 1000
ALLOWED_DOMAINS = ["docs.example.com"]
EXCLUDED_PATHS = ["/api/", "/static/"]

# Chunking Settings
SMALL_CHUNK_SIZE = 500      # characters
LARGE_CHUNK_SIZE = 2000     # characters
CHUNK_OVERLAP = 100         # characters

# Search Settings
BM25_K1 = 1.5
BM25_B = 0.75
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3
TOP_K_RETRIEVAL = 20
TOP_K_FINAL = 5
MAX_CHUNKS_PER_PAGE = 2

# Confidence Threshold
MIN_CONFIDENCE_SCORE = 3  # Below this, say "I don't know"
```

---

## Performance Expectations

For a corpus of ~500 pages:

| Operation | Expected Time |
|-----------|---------------|
| Full crawl | 10-30 minutes (depends on rate limiting) |
| Full indexing | 30-60 minutes (LLM calls are slow) |
| Query response | 3-10 seconds |

### Accuracy Expectations

With all optimizations implemented:
- **Simple factual queries:** 85-90% accuracy
- **How-to queries:** 75-85% accuracy
- **Global queries:** 70-80% accuracy
- **Overall:** ~80-85% useful answers

This is significantly better than basic RAG (~60-70%) but won't reach enterprise solutions like Glean (~90-95%) which have years of engineering, user behavior signals, and massive infrastructure.

---

## Dependencies

**Required:** Python 3.6.5+ standard library only

**Standard library modules used:**
- `urllib.request`, `urllib.parse` - HTTP requests
- `html.parser` - HTML parsing
- `sqlite3` - Database
- `json` - Data serialization
- `re` - Text processing
- `math` - Cosine similarity, BM25
- `hashlib` - Content hashing
- `logging` - Logging
- `argparse` - CLI parsing
- `collections` - Counter, defaultdict
- `ssl` - SSL configuration

**External (via your custom APIs):**
- Vector embedding API
- Chat/LLM completion API
