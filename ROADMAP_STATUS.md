# Roadmap Status

**Last Updated:** 2026-01-28

This file tracks implementation progress against the GitHub issues roadmap. Use this to quickly understand what's done and what's left.

## Summary

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | **COMPLETE** | Robustness & Error Handling |
| Phase 2 | **COMPLETE** | Performance & Observability |
| Phase 3 | **NOT STARTED** | Feature Completeness (Knowledge Graph, Summarization) |
| Phase 4 | **PARTIAL** | Documentation & Deployment |

## Current State

The RAG system is **production-ready for basic use cases**:
- Crawling, chunking, and indexing works
- Hybrid search (BM25 + vector) works
- Query expansion and answer generation works
- OpenAI integration works out of the box
- BM25-only mode works without any AI/API dependencies

**What's NOT wired up yet:**
- Entity extraction exists but isn't called during indexing
- Page/system/global summarization exists but isn't called
- Knowledge graph exists but isn't used in answer generation
- Grounding safeguards (citation validation) not implemented

## Phase Details

### Phase 1: Robustness (COMPLETE)

All issues #13-#19 are closed:
- [x] #13 - Configuration validation with type hints
- [x] #14 - Database error handling and transactions
- [x] #15 - API client error handling and circuit breaker
- [x] #16 - Crawler error handling and URL validation
- [x] #17 - Security hardening
- [x] #18 - Graceful shutdown and resource cleanup
- [x] #19 - Health check and startup validation

### Phase 2: Performance (COMPLETE)

All issues #20-#24 are closed:
- [x] #20 - Structured logging and metrics collection
- [x] #21 - Vector search performance optimization
- [x] #22 - BM25 batch query optimization
- [x] #23 - Query result caching (LRU + TTL)
- [x] #24 - Testing improvements (E2E, benchmarks)

### Phase 3: Feature Completeness (NOT STARTED)

Issues #25-#29 are all open:
- [ ] #25 - Wire entity extractor into indexer pipeline
- [ ] #26 - Wire page summarizer into indexer pipeline
- [ ] #27 - Generate system and global summaries automatically
- [ ] #28 - Add knowledge graph context to answer generation
- [ ] #29 - Add missing CLI commands

**CLI Commands Status:**
| Command | Status | Notes |
|---------|--------|-------|
| `rebuild-index` | EXISTS | Rebuilds BM25 index |
| `cache stats/clear` | EXISTS | Query cache management |
| `rebuild-summaries` | MISSING | |
| `export-graph` | MISSING | Export knowledge graph as JSON |
| `validate` | MISSING | Validate database integrity |

### Phase 4: Documentation & Deployment (PARTIAL)

Issues #30-#33:
- [x] #30 - Type hints in config.py - **ALREADY DONE, can close**
- [ ] #31 - Split utils.py into focused modules
- [ ] #32 - Deployment artifacts (partial - see below)
- [ ] #33 - Documentation

**Deployment Artifacts Status:**
| Artifact | Status |
|----------|--------|
| `.env.example` | EXISTS |
| GitHub Actions | EXISTS (claude.yml, claude-code-review.yml) |
| `Dockerfile` | MISSING |
| `docker-compose.yml` | MISSING |
| `Makefile` | MISSING |
| `/docs` directory | MISSING |

### Standalone Issues

- [ ] #49 - Grounding safeguards to prevent hallucination
- [x] #50 - Crawler resume capability (CLOSED)

## Recent Changes (2026-01-28)

- Merged `feature/contextual-retrieval` - adds contextual embeddings (chunk + title + heading path)
- Merged `fix/md5-to-sha256-fips-compliance` - FIPS compliance for hashing
- Added `--force` flag to `backfill` command for embedding regeneration
- Added `scripts/regenerate-embeddings.sh` convenience script

## Quick Reference: What Works Today

```bash
# Crawl a site
python -m rag_system.main ingest https://docs.example.com --max-pages 100

# Query with AI (needs OPENAI_API_KEY)
python -m rag_system.main query "How do I configure X?"

# Query without AI (BM25 only)
RAG_AI_ENABLED=false python -m rag_system.main query "configure timeout"

# Or use pure BM25 search
python -m rag_system.main bm25 "configure timeout"

# Regenerate embeddings with contextual retrieval
python -m rag_system.main backfill --force

# Health check
python -m rag_system.main health

# Stats
python -m rag_system.main stats
```

## Next Steps (Suggested Priority)

1. Close #30 (already done)
2. Update #32 to mark completed items
3. Implement #49 (grounding safeguards) - high value, prevents hallucination
4. Implement #25-#28 (wire up entity extraction, summarization, knowledge graph)
5. Implement remaining CLI commands (#29)
6. Add deployment artifacts (#32) and docs (#33)
