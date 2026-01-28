"""Main CLI module for the RAG system.

Provides command-line interface for interacting with the RAG system.
"""

import argparse
import hashlib
import os
from typing import Dict, List, Any, Tuple, Optional


def load_dotenv(path: str = '.env') -> None:
    """Load environment variables from a .env file.

    Args:
        path: Path to .env file. Defaults to '.env' in current directory.
    """
    if not os.path.exists(path):
        return

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Don't override existing env vars
                if key and key not in os.environ:
                    os.environ[key] = value


# Load .env file at import time (before config is loaded elsewhere)
load_dotenv()

from rag_system import config
from rag_system.database import (
    init_db, get_connection, managed_connection, cache_query, get_cached_query, log_query,
    get_query_analytics, export_query_logs_csv, migrate_query_log_timing
)
from rag_system.api_client import (
    VectorAPIClient, ChatAPIClient,
    create_openai_vector_client, create_openai_chat_client,
    RateLimitError
)
from rag_system.security import (
    validate_query_length_or_raise, ValidationError,
    MAX_QUERY_LENGTH, sanitize_for_logging
)
from rag_system.shutdown import (
    install_shutdown_handlers, register_cleanup, is_shutdown_requested
)
from rag_system.health import HealthChecker, format_health_report
from rag_system.search.bm25_search import BM25Search
from rag_system.search.vector_search import VectorSearch
from rag_system.search.hybrid_search import HybridSearch
from rag_system.search.reranker import Reranker
from rag_system.search.diversifier import Diversifier
from rag_system.query.classifier import QueryClassifier
from rag_system.query.expander import QueryExpander
from rag_system.query.confidence import ConfidenceAnalyzer
from rag_system.query.context_builder import ContextBuilder
from rag_system.query.answer_generator import AnswerGenerator
from rag_system.utils import get_logger, get_metrics_collector, Timer, LRUCache

logger = get_logger(__name__)
metrics_collector = get_metrics_collector()

# Global query cache instance (initialized lazily by RAGSystem)
_query_cache: Optional[LRUCache] = None


def get_query_cache() -> Optional[LRUCache]:
    """Get the global query cache instance.

    Returns:
        LRUCache instance or None if caching is disabled.
    """
    return _query_cache


def parse_command(input_str: str) -> Tuple[str, List[str]]:
    """Parse a command string into command and arguments.

    Args:
        input_str: Raw input string.

    Returns:
        Tuple of (command, args_list).
    """
    parts = input_str.strip().split(maxsplit=1)
    if not parts:
        return '', []

    cmd = parts[0].lower()
    args = [parts[1]] if len(parts) > 1 else []
    return cmd, args


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description='Intelligent Documentation RAG System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--db', default=config.DATABASE_PATH,
        help='Path to SQLite database'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documentation')
    ingest_parser.add_argument('url', help='Starting URL to crawl')
    ingest_parser.add_argument(
        '--max-pages', type=int, default=config.MAX_PAGES,
        help='Maximum pages to crawl'
    )
    ingest_parser.add_argument(
        '--unlimited', action='store_true',
        help='Crawl all pages with no limit (overrides --max-pages)'
    )
    ingest_parser.add_argument(
        '--ignore-robots', action='store_true',
        help='Ignore robots.txt restrictions (use responsibly)'
    )
    ingest_parser.add_argument(
        '--fresh', action='store_true',
        help='Start a new crawl even if a resumable session exists'
    )
    ingest_parser.add_argument(
        '--basic-auth-user',
        help='Username for HTTP Basic Auth'
    )
    ingest_parser.add_argument(
        '--basic-auth-pass',
        help='Password for HTTP Basic Auth'
    )
    ingest_parser.add_argument(
        '--basic-auth-token',
        help='Pre-encoded Base64 token for HTTP Basic Auth (overrides user/pass)'
    )

    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument(
        '--top-k', type=int, default=config.TOP_K_FINAL,
        help='Number of results to return'
    )

    # BM25 search command (no AI required)
    bm25_parser = subparsers.add_parser('bm25', help='BM25 lexical search (no AI)')
    bm25_parser.add_argument('query', help='Search query')
    bm25_parser.add_argument(
        '--top-k', type=int, default=config.TOP_K_FINAL,
        help='Number of results to return'
    )
    bm25_parser.add_argument(
        '--json', action='store_true',
        help='Output results as JSON'
    )

    # Rebuild BM25 index command
    subparsers.add_parser('rebuild-index', help='Rebuild the BM25 search index')

    # Extract entities command
    extract_entities_parser = subparsers.add_parser(
        'extract-entities',
        help='Extract entities from pages (post-processing step)'
    )
    extract_entities_parser.add_argument(
        '--page-id', type=int,
        help='Only extract entities for a specific page ID'
    )
    extract_entities_parser.add_argument(
        '--json', action='store_true',
        help='Output results as JSON'
    )

    # Summarize pages command
    summarize_pages_parser = subparsers.add_parser(
        'summarize-pages',
        help='Generate summaries for pages (post-processing step)'
    )
    summarize_pages_parser.add_argument(
        '--page-id', type=int,
        help='Only summarize a specific page ID'
    )
    summarize_pages_parser.add_argument(
        '--force', action='store_true',
        help='Regenerate summaries even for pages that have them'
    )
    summarize_pages_parser.add_argument(
        '--json', action='store_true',
        help='Output results as JSON'
    )

    # Rebuild summaries command (system + global)
    rebuild_summaries_parser = subparsers.add_parser(
        'rebuild-summaries',
        help='Rebuild system and global summaries (post-processing step)'
    )
    rebuild_summaries_parser.add_argument(
        '--json', action='store_true',
        help='Output results as JSON'
    )

    # Stats command
    subparsers.add_parser('stats', help='Show system statistics')

    # Interactive command
    subparsers.add_parser('interactive', help='Start interactive mode')

    # Health command
    subparsers.add_parser('health', help='Check system health')

    # Analytics command
    analytics_parser = subparsers.add_parser('analytics', help='Show query analytics')
    analytics_parser.add_argument(
        '--export', type=str, metavar='FILE',
        help='Export query logs to CSV file'
    )
    analytics_parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit number of rows to export'
    )
    analytics_parser.add_argument(
        '--json', action='store_true',
        help='Output analytics as JSON'
    )

    # Cache command
    cache_parser = subparsers.add_parser('cache', help='Manage query cache')
    cache_parser.add_argument(
        'action', choices=['stats', 'clear'],
        help='Cache action: stats (show statistics) or clear (clear cache)'
    )
    cache_parser.add_argument(
        '--json', action='store_true',
        help='Output as JSON'
    )

    # Backfill embeddings command
    backfill_parser = subparsers.add_parser(
        'backfill',
        help='Generate embeddings for chunks that are missing them'
    )
    backfill_parser.add_argument(
        '--batch-size', type=int, default=config.EMBEDDING_BATCH_SIZE,
        help='Number of chunks to embed per API call'
    )
    backfill_parser.add_argument(
        '--force', action='store_true',
        help='Regenerate all embeddings, even for chunks that already have them'
    )

    # Crawl sessions command
    sessions_parser = subparsers.add_parser(
        'crawl-sessions',
        help='View and manage crawl sessions'
    )
    sessions_parser.add_argument(
        '--json', action='store_true',
        help='Output as JSON'
    )

    # Resume embeddings command
    resume_parser = subparsers.add_parser(
        'resume-embeddings',
        help='Resume embedding generation for chunks without embeddings'
    )
    resume_parser.add_argument(
        '--page-id', type=int,
        help='Only embed chunks for a specific page ID'
    )
    resume_parser.add_argument(
        '--batch-size', type=int, default=config.EMBEDDING_BATCH_SIZE,
        help='Number of chunks to embed per API call'
    )

    return parser


def format_query_result(result: Dict[str, Any]) -> str:
    """Format query result for display.

    Args:
        result: Query result dict.

    Returns:
        Formatted string.
    """
    lines = []

    # Answer
    answer = result.get('answer', 'No answer available.')
    lines.append(f"Answer: {answer}")
    lines.append("")

    # Confidence
    confidence = result.get('confidence', 0)
    confidence_pct = int(confidence * 100)
    lines.append(f"Confidence: {confidence_pct}%")

    # Sources - deduplicated by URL, showing page title and URL
    sources = result.get('sources', [])
    if sources:
        lines.append("")
        lines.append("Sources:")
        for source in sources:
            title = source.get('page_title', 'Untitled')
            url = source.get('page_url', '')
            lines.append(f"  \u2022 {title}")
            if url:
                lines.append(f"    {url}")

    return '\n'.join(lines)


def format_stats(stats: Dict[str, Any]) -> str:
    """Format statistics for display.

    Args:
        stats: Statistics dict.

    Returns:
        Formatted string.
    """
    lines = ["RAG System Statistics", "=" * 30]

    for key, value in stats.items():
        lines.append(f"{key.capitalize()}: {value}")

    return '\n'.join(lines)


class RAGSystem:
    """Main RAG system class."""

    def __init__(self, db_path: str = None,
                 vector_client: Optional[VectorAPIClient] = None,
                 chat_client: Optional[ChatAPIClient] = None):
        """Initialize the RAG system.

        Args:
            db_path: Path to SQLite database.
            vector_client: Optional vector API client. If None and OPENAI_API_KEY
                is set, an OpenAI client will be created automatically.
            chat_client: Optional chat API client. If None and OPENAI_API_KEY
                is set, an OpenAI client will be created automatically.
        """
        self.db_path = db_path or config.DATABASE_PATH

        # Auto-create OpenAI clients if API key is available and no clients provided
        # Only create AI clients if AI is enabled
        if config.AI_ENABLED and vector_client is None and chat_client is None:
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                logger.info("Using OpenAI API for embeddings and chat")
                vector_client = create_openai_vector_client()
                chat_client = create_openai_chat_client()
        elif not config.AI_ENABLED:
            logger.info("AI features disabled (RAG_AI_ENABLED=false), using BM25 search only")

        self.vector_client = vector_client
        self.chat_client = chat_client

        # Initialize database
        init_db(self.db_path)

        # Initialize components
        self.classifier = QueryClassifier(chat_client)
        self.expander = QueryExpander(chat_client)
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.context_builder = ContextBuilder(self.db_path)
        self.answer_generator = AnswerGenerator(chat_client)

        # Initialize search components
        self.bm25_search = BM25Search(self.db_path)
        self.vector_search = VectorSearch(self.db_path, vector_client)
        self.hybrid_search = HybridSearch(self.vector_search, self.bm25_search)
        self.reranker = Reranker(self.db_path)
        self.diversifier = Diversifier(self.db_path)

        # Initialize query result cache
        global _query_cache
        if config.QUERY_CACHE_ENABLED:
            _query_cache = LRUCache(
                max_size=config.QUERY_CACHE_SIZE,
                ttl_seconds=config.QUERY_CACHE_TTL
            )
            logger.info(f"Query cache enabled: size={config.QUERY_CACHE_SIZE}, ttl={config.QUERY_CACHE_TTL}s")
        else:
            _query_cache = None
            logger.info("Query cache disabled")

    def query(self, question: str, top_k: int = None,
              use_cache: bool = True) -> Dict[str, Any]:
        """Query the system.

        Args:
            question: Question to ask.
            top_k: Number of results.
            use_cache: Whether to use the query result cache.

        Returns:
            Result dict with answer, chunks, confidence, and timing metrics.

        Raises:
            ValidationError: If query exceeds maximum length.
        """
        import time
        query_start_time = time.time()
        timing_metrics: Dict[str, float] = {}

        # Validate query length for security
        validate_query_length_or_raise(question)

        top_k = top_k or config.TOP_K_FINAL

        # Generate query hash for caching (SHA256 for FIPS compliance)
        query_hash = hashlib.sha256(question.lower().strip().encode()).hexdigest()

        # Check query result cache first
        cache_key = f"{query_hash}:{top_k}"
        if use_cache and _query_cache is not None:
            cached_result = _query_cache.get(cache_key)
            if cached_result is not None:
                # Add cache hit info to result
                cached_result = cached_result.copy()
                cached_result['cache_hit'] = True
                cached_result['timing'] = {'total_time': time.time() - query_start_time}
                logger.debug(f"Cache hit for query: {sanitize_for_logging(question, 50)}")
                return cached_result

        # Check cache for query processing results
        conn = get_connection(self.db_path)
        cached = None
        try:
            cached = get_cached_query(conn, query_hash)
        finally:
            conn.close()

        if cached:
            query_type = cached.get('query_type', 'factual')
            import json
            expanded = json.loads(cached.get('expanded_queries', '[]')) or [question]
            logger.info(f"Using cached query processing for: {sanitize_for_logging(question, 50)}")
        else:
            # Classify query
            query_type = self.classifier.classify(question)
            # Expand query
            expanded = self.expander.expand(question)

        # Search - use BM25 only if no vector client, otherwise hybrid
        all_results = []
        search_start = time.time()

        # Batch embed all expanded queries at once for performance
        query_embeddings: List[Optional[List[float]]] = []
        if config.AI_ENABLED and self.vector_client and expanded:
            try:
                embed_start = time.time()
                query_embeddings = self.vector_client.get_embeddings_batch(expanded)
                timing_metrics['embedding_time'] = time.time() - embed_start
            except Exception as e:
                logger.warning(f"Vector embedding failed, using BM25 only: {e}")
                query_embeddings = [None] * len(expanded)
        else:
            query_embeddings = [None] * len(expanded)

        for q, query_embedding in zip(expanded, query_embeddings):
            # Hybrid search handles None embedding gracefully (returns BM25 only)
            search_op_start = time.time()
            results = self.hybrid_search.search(query_embedding, q)
            timing_metrics['search_time'] = timing_metrics.get('search_time', 0) + (time.time() - search_op_start)

            all_results.extend(results)

        # Deduplicate by chunk_id
        seen = set()
        unique_results = []
        for chunk_id, score in all_results:
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_results.append((chunk_id, score))

        # Rerank
        rerank_start = time.time()
        reranked = self.reranker.rerank(unique_results, question)
        timing_metrics['rerank_time'] = time.time() - rerank_start

        # Diversify
        diversified = self.diversifier.diversify(reranked, top_k=top_k)

        # Build chunk info with page metadata for citations
        chunks = []
        conn = get_connection(self.db_path)
        try:
            for chunk_id, score in diversified:
                cursor = conn.execute(
                    """SELECT c.content, c.heading_path, p.title as page_title, p.url as page_url
                       FROM chunks c JOIN pages p ON c.page_id = p.id WHERE c.id = ?""",
                    (chunk_id,)
                )
                row = cursor.fetchone()
                if row:
                    chunks.append({
                        'id': chunk_id,
                        'score': score,
                        'content': row['content'],
                        'heading_path': row['heading_path'],
                        'page_title': row['page_title'],
                        'page_url': row['page_url']
                    })
        finally:
            conn.close()

        # Calculate confidence
        metrics = self.confidence_analyzer.analyze(question, chunks)

        # Build context and generate answer with grounding safeguards
        generation_start = time.time()
        citations_valid = True
        valid_citations = []
        invalid_citations = []
        used_fallback = False

        if chunks and self.chat_client:
            # Use grounded generation with citation validation
            answer_result = self.answer_generator.generate_grounded(
                question, chunks, confidence=metrics['overall']
            )
            answer = answer_result['answer']
            citations_valid = answer_result.get('citations_valid', True)
            valid_citations = answer_result.get('valid_citations', [])
            invalid_citations = answer_result.get('invalid_citations', [])
            used_fallback = answer_result.get('used_fallback', False)

            # Log warning if citations were fabricated
            if not citations_valid:
                logger.warning(
                    f"Answer contained {len(invalid_citations)} fabricated citations: "
                    f"{invalid_citations}"
                )

            timing_metrics['generation_time'] = time.time() - generation_start
        else:
            answer = "No relevant information found." if not chunks else \
                     "Answer generation is not configured."

        # Cache query processing if not already cached
        if not cached:
            conn = get_connection(self.db_path)
            try:
                # Reuse first embedding from batch (already computed above)
                first_query_embedding = query_embeddings[0] if query_embeddings else []
                cache_query(conn, query_hash, query_type, expanded, first_query_embedding or [])
            finally:
                conn.close()

        # Calculate total time
        timing_metrics['total_time'] = time.time() - query_start_time

        # Log the query with timing metrics
        conn = get_connection(self.db_path)
        try:
            chunk_ids = [c['id'] for c in chunks]
            log_query(
                conn, question, query_type, expanded, chunk_ids,
                metrics['overall'], bool(chunks and self.chat_client),
                metrics=timing_metrics
            )
        finally:
            conn.close()

        # Record metrics to global collector
        for name, value in timing_metrics.items():
            metrics_collector.record(name, value)

        # Build deduplicated sources list for citations
        seen_urls = set()
        sources = []
        for chunk in chunks:
            url = chunk.get('page_url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    'page_title': chunk.get('page_title', 'Untitled'),
                    'page_url': url
                })

        result = {
            'query': question,
            'query_type': query_type,
            'answer': answer,
            'chunks': chunks,
            'sources': sources,
            'confidence': metrics['overall'],
            'metrics': metrics,
            'timing': timing_metrics,
            'cache_hit': False,
            'citations_valid': citations_valid,
            'valid_citations': valid_citations,
            'invalid_citations': invalid_citations,
            'used_fallback': used_fallback
        }

        # Log query timing (WARNING if >2s, INFO otherwise)
        total_time = timing_metrics.get('total_time', 0)
        if total_time > 2.0:
            logger.warning(f"Query completed in {total_time:.2f}s (slow)")
        else:
            logger.info(f"Query completed in {total_time:.2f}s")

        # Store result in cache
        if use_cache and _query_cache is not None:
            _query_cache.put(cache_key, result)
            logger.debug(f"Cached result for query: {sanitize_for_logging(question, 50)}")

        return result

    def ingest(self, start_url: str, max_pages: int = None,
                ignore_robots: bool = False, fresh: bool = False,
                basic_auth_token: Optional[str] = None) -> Dict[str, Any]:
        """Ingest documentation from URL.

        Supports resuming interrupted crawls. If a previous crawl for the same
        URL was interrupted, it will automatically resume from where it left off
        unless fresh=True is specified.

        Args:
            start_url: Starting URL to crawl.
            max_pages: Maximum pages to crawl.
            ignore_robots: If True, ignore robots.txt restrictions.
            fresh: If True, start a new crawl even if a resumable session exists.
            basic_auth_token: Optional Base64-encoded token for HTTP Basic Auth.

        Returns:
            Ingestion statistics.
        """
        from rag_system.ingestion.indexer import Indexer
        from urllib.parse import urlparse

        max_pages = max_pages or config.MAX_PAGES

        # Extract domain from start URL
        parsed = urlparse(start_url)
        allowed_domains = config.ALLOWED_DOMAINS or [parsed.netloc]

        indexer = Indexer(
            self.db_path,
            vector_client=self.vector_client,
            chat_client=self.chat_client
        )

        return indexer.crawl_and_index(
            start_url=start_url,
            allowed_domains=allowed_domains,
            excluded_paths=config.EXCLUDED_PATHS,
            included_paths=config.INCLUDED_PATHS,
            max_pages=max_pages,
            delay=config.CRAWL_DELAY_SECONDS,
            ignore_robots=ignore_robots,
            fresh=fresh,
            basic_auth_token=basic_auth_token
        )

    def get_stats(self) -> Dict[str, int]:
        """Get system statistics.

        Returns:
            Statistics dict.
        """
        conn = get_connection(self.db_path)
        try:
            stats = {}

            # Count pages
            cursor = conn.execute("SELECT COUNT(*) as count FROM pages")
            stats['pages'] = cursor.fetchone()['count']

            # Count chunks
            cursor = conn.execute("SELECT COUNT(*) as count FROM chunks")
            stats['chunks'] = cursor.fetchone()['count']

            # Count entities
            cursor = conn.execute("SELECT COUNT(*) as count FROM entities")
            stats['entities'] = cursor.fetchone()['count']

            # Count relationships
            cursor = conn.execute("SELECT COUNT(*) as count FROM relationships")
            stats['relationships'] = cursor.fetchone()['count']

            return stats
        finally:
            conn.close()

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get query cache statistics.

        Returns:
            Dict with cache stats, or None if caching is disabled.
        """
        if _query_cache is None:
            return None
        return _query_cache.get_stats()

    def clear_cache(self) -> bool:
        """Clear the query result cache.

        Returns:
            True if cache was cleared, False if caching is disabled.
        """
        if _query_cache is None:
            return False
        _query_cache.clear()
        logger.info("Query result cache cleared")
        return True

    def invalidate_cache(self) -> bool:
        """Invalidate cache entries (call when database changes).

        Returns:
            True if cache was invalidated, False if caching is disabled.
        """
        return self.clear_cache()


def run_interactive(rag: RAGSystem) -> None:
    """Run interactive mode.

    Args:
        rag: RAGSystem instance.
    """
    print("RAG System Interactive Mode")
    print("Type 'quit' or 'exit' to quit, 'help' for commands")
    print()

    while True:
        try:
            user_input = input("rag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd, args = parse_command(user_input)

        if cmd in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        elif cmd == 'help':
            print("Commands:")
            print("  query <question>  - Ask a question")
            print("  stats             - Show statistics")
            print("  cache             - Show cache statistics")
            print("  cache clear       - Clear the cache")
            print("  quit/exit         - Exit")
        elif cmd == 'stats':
            stats = rag.get_stats()
            print(format_stats(stats))
        elif cmd == 'cache':
            if args and args[0] == 'clear':
                if rag.clear_cache():
                    print("Cache cleared")
                else:
                    print("Caching is disabled")
            else:
                cache_stats = rag.get_cache_stats()
                if cache_stats is None:
                    print("Caching is disabled")
                else:
                    print(f"Cache: {cache_stats['size']}/{cache_stats['max_size']} entries, "
                          f"hit rate: {cache_stats['hit_rate']:.1%}")
        elif cmd == 'query' and args:
            result = rag.query(args[0])
            print(format_query_result(result))
        else:
            # Treat as query
            result = rag.query(user_input)
            print(format_query_result(result))

        print()


def backfill_embeddings(db_path: str, vector_client: Any,
                         batch_size: int = 100, force: bool = False) -> Dict[str, int]:
    """Generate embeddings for chunks that don't have them.

    Args:
        db_path: Path to SQLite database.
        vector_client: Vector API client for embeddings.
        batch_size: Number of chunks to embed per API call.
        force: If True, regenerate embeddings for all chunks, not just missing ones.

    Returns:
        Dict with statistics about the backfill operation.
    """
    import time
    from rag_system.database import get_connection, update_chunk_embedding
    from rag_system.ingestion.indexer import build_contextual_text
    from rag_system.api_client import APIError

    conn = get_connection(db_path)
    try:
        # Find chunks to embed (all chunks if force, otherwise only missing)
        if force:
            cursor = conn.execute("""
                SELECT c.id, c.content, c.heading_path, p.title
                FROM chunks c
                JOIN pages p ON c.page_id = p.id
                ORDER BY c.id
            """)
        else:
            cursor = conn.execute("""
                SELECT c.id, c.content, c.heading_path, p.title
                FROM chunks c
                JOIN pages p ON c.page_id = p.id
                WHERE c.embedding_json IS NULL
                ORDER BY c.id
            """)
        chunks_to_embed = cursor.fetchall()

        total = len(chunks_to_embed)
        if total == 0:
            print("No chunks found to embed.")
            return {'total': 0, 'embedded': 0, 'errors': 0}

        if force:
            print(f"Regenerating embeddings for {total} chunks (force mode)")
        else:
            print(f"Found {total} chunks without embeddings")

        embedded = 0
        errors = 0
        batch_delay = config.EMBEDDING_BATCH_DELAY

        for i in range(0, total, batch_size):
            batch = chunks_to_embed[i:i + batch_size]

            # Build contextual text for each chunk
            texts = []
            for row in batch:
                chunk = {
                    'content': row['content'],
                    'heading_path': row['heading_path']
                }
                texts.append(build_contextual_text(chunk, row['title']))

            try:
                embeddings = vector_client.get_embeddings_batch(texts)

                for row, embedding in zip(batch, embeddings):
                    update_chunk_embedding(conn, row['id'], embedding)
                    embedded += 1

                print(f"Progress: {embedded}/{total} chunks embedded")

            except APIError as e:
                logger.error(f"API error during backfill: {e}")
                errors += len(batch)
            except Exception as e:
                logger.error(f"Unexpected error during backfill: {e}")
                errors += len(batch)

            # Rate limit between batches
            if i + batch_size < total and batch_delay > 0:
                time.sleep(batch_delay)

        print(f"Backfill complete: {embedded} embedded, {errors} errors")
        return {'total': total, 'embedded': embedded, 'errors': errors}

    finally:
        conn.close()


def main() -> None:
    """Main entry point."""
    # Install graceful shutdown handlers
    install_shutdown_handlers()

    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    rag = None
    try:
        rag = RAGSystem(db_path=args.db)

        if args.command == 'ingest':
            import sys
            max_pages = sys.maxsize if args.unlimited else args.max_pages
            if args.unlimited:
                print(f"Ingesting from {args.url} (unlimited pages)...")
            else:
                print(f"Ingesting from {args.url} (max {max_pages} pages)...")
            if args.fresh:
                print("Starting fresh crawl (ignoring any existing session)...")

            # Get Basic Auth token from CLI args or config
            basic_auth_token = None
            cli_token = getattr(args, 'basic_auth_token', None)
            cli_user = getattr(args, 'basic_auth_user', None)
            cli_pass = getattr(args, 'basic_auth_pass', None)

            if cli_token or cli_user or cli_pass:
                # CLI args take precedence
                basic_auth_token = config.get_basic_auth_token(
                    username=cli_user,
                    password=cli_pass,
                    token=cli_token
                )
                if basic_auth_token:
                    print("Using HTTP Basic Auth from command line...")
            elif config.BASIC_AUTH_ENABLED:
                # Fall back to config/environment variables
                basic_auth_token = config.get_basic_auth_token()
                if basic_auth_token:
                    print("Using HTTP Basic Auth from environment...")

            try:
                stats = rag.ingest(
                    args.url,
                    max_pages=max_pages,
                    ignore_robots=args.ignore_robots,
                    fresh=args.fresh,
                    basic_auth_token=basic_auth_token
                )
                resumed_msg = " (resumed)" if stats.get('resumed') else ""
                print(f"Crawled {stats['pages_crawled']} pages{resumed_msg}, indexed {stats['pages_indexed']}, skipped {stats['pages_skipped']}, errors: {stats['errors']}")
            except KeyboardInterrupt:
                print("\nIngestion interrupted. Progress has been saved - run the same command to resume.")
                logger.info("Ingestion interrupted by user")

        elif args.command == 'query':
            # When AI is disabled, fall back to BM25 search
            if not config.AI_ENABLED:
                print("AI features disabled. Showing BM25 search results:\n")
                results = rag.bm25_search.search(args.question, top_k=args.top_k)
                if not results:
                    print("No results found.")
                else:
                    print(f"Search Results ({len(results)} matches)")
                    print("=" * 60)
                    conn = get_connection(args.db)
                    try:
                        from rag_system.database import get_chunk_by_id
                        for i, (chunk_id, score) in enumerate(results, 1):
                            chunk = get_chunk_by_id(conn, chunk_id)
                            if chunk:
                                heading = chunk['heading_path'] or 'No heading'
                                content = chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content']
                                print(f"\n{i}. [{heading}] (score: {score:.4f})")
                                print(f"   {content}")
                    finally:
                        conn.close()
            else:
                result = rag.query(args.question, top_k=args.top_k)
                print(format_query_result(result))

        elif args.command == 'bm25':
            # Pure BM25 search - no AI, no query expansion, just lexical matching
            results = rag.bm25_search.search(args.query, top_k=args.top_k)

            if args.json:
                import json as json_module
                # Fetch chunk details for JSON output
                conn = get_connection(args.db)
                try:
                    from rag_system.database import get_chunk_by_id
                    output = []
                    for chunk_id, score in results:
                        chunk = get_chunk_by_id(conn, chunk_id)
                        if chunk:
                            output.append({
                                'chunk_id': chunk_id,
                                'score': round(score, 4),
                                'content': chunk['content'],
                                'heading_path': chunk['heading_path'],
                                'url': chunk.get('url', '')
                            })
                    print(json_module.dumps(output, indent=2))
                finally:
                    conn.close()
            else:
                # Human-readable output
                if not results:
                    print("No results found.")
                else:
                    print(f"BM25 Search Results ({len(results)} matches)")
                    print("=" * 60)
                    conn = get_connection(args.db)
                    try:
                        from rag_system.database import get_chunk_by_id
                        for i, (chunk_id, score) in enumerate(results, 1):
                            chunk = get_chunk_by_id(conn, chunk_id)
                            if chunk:
                                heading = chunk['heading_path'] or 'No heading'
                                content = chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content']
                                print(f"\n{i}. [{heading}] (score: {score:.4f})")
                                print(f"   {content}")
                    finally:
                        conn.close()

        elif args.command == 'stats':
            stats = rag.get_stats()
            print(format_stats(stats))

        elif args.command == 'interactive':
            run_interactive(rag)

        elif args.command == 'health':
            checker = HealthChecker(
                db_path=args.db,
                vector_client=rag.vector_client,
                chat_client=rag.chat_client
            )
            result = checker.check_all()
            print(format_health_report(result))

            # Exit with non-zero if unhealthy
            if not result['healthy']:
                import sys
                sys.exit(1)

        elif args.command == 'analytics':
            import json as json_module
            conn = get_connection(args.db)
            try:
                # Run schema migration for timing columns
                migrate_query_log_timing(conn)

                if args.export:
                    # Export to CSV
                    count = export_query_logs_csv(conn, args.export, args.limit)
                    print(f"Exported {count} query logs to {args.export}")
                else:
                    # Show analytics
                    analytics = get_query_analytics(conn)
                    if args.json:
                        print(json_module.dumps(analytics, indent=2))
                    else:
                        print("Query Analytics")
                        print("=" * 40)
                        print(f"Total queries: {analytics['total_queries']}")
                        print(f"Queries with answers: {analytics['queries_with_answers']}")
                        print(f"Average confidence: {analytics['avg_confidence']:.2%}")
                        print()
                        print("Query Types:")
                        for qtype, count in analytics['query_types'].items():
                            print(f"  {qtype}: {count}")
                        print()
                        print("Timing (average ms):")
                        timing = analytics['timing']
                        if timing['avg_total_ms']:
                            print(f"  Total: {timing['avg_total_ms']:.1f}ms")
                        if timing['avg_embedding_ms']:
                            print(f"  Embedding: {timing['avg_embedding_ms']:.1f}ms")
                        if timing['avg_search_ms']:
                            print(f"  Search: {timing['avg_search_ms']:.1f}ms")
                        if timing['avg_rerank_ms']:
                            print(f"  Rerank: {timing['avg_rerank_ms']:.1f}ms")
                        if timing['avg_generation_ms']:
                            print(f"  Generation: {timing['avg_generation_ms']:.1f}ms")
            finally:
                conn.close()

        elif args.command == 'cache':
            import json as json_module
            cache_stats = rag.get_cache_stats()

            if args.action == 'stats':
                if cache_stats is None:
                    print("Query caching is disabled")
                elif args.json:
                    print(json_module.dumps(cache_stats, indent=2))
                else:
                    print("Query Cache Statistics")
                    print("=" * 40)
                    print(f"Size: {cache_stats['size']} / {cache_stats['max_size']}")
                    print(f"TTL: {cache_stats['ttl_seconds']} seconds")
                    print(f"Hits: {cache_stats['hits']}")
                    print(f"Misses: {cache_stats['misses']}")
                    print(f"Evictions: {cache_stats['evictions']}")
                    print(f"Hit Rate: {cache_stats['hit_rate']:.1%}")
            elif args.action == 'clear':
                if rag.clear_cache():
                    print("Cache cleared successfully")
                else:
                    print("Query caching is disabled")

        elif args.command == 'rebuild-index':
            from rag_system.search.bm25_search import BM25Index
            print("Rebuilding BM25 search index...")
            bm25_index = BM25Index(args.db)
            bm25_index.build()
            # Verify it worked
            conn = get_connection(args.db)
            try:
                from rag_system.database import get_corpus_stats
                stats = get_corpus_stats(conn)
                if stats:
                    print(f"Index rebuilt: {stats['total_docs']} documents indexed")
                else:
                    print("Warning: Index may not have built correctly")
            finally:
                conn.close()

        elif args.command == 'extract-entities':
            if not rag.chat_client:
                print("Error: No chat client configured.")
                print("Set OPENAI_API_KEY or configure RAG_CHAT_API_ENDPOINT")
                import sys
                sys.exit(1)

            from rag_system.ingestion.indexer import Indexer
            import json as json_module

            print("Extracting entities from pages...")
            indexer = Indexer(args.db, chat_client=rag.chat_client)
            stats = indexer.extract_entities(page_id=args.page_id)

            if args.json:
                print(json_module.dumps(stats, indent=2))
            else:
                if stats.get('error'):
                    print(f"Error: {stats['error']}")
                else:
                    print("Entity Extraction Results")
                    print("=" * 40)
                    print(f"Pages processed: {stats['pages_processed']}")
                    print(f"Entities extracted: {stats['entities_extracted']}")
                    print(f"Relationships extracted: {stats['relationships_extracted']}")
                    if stats['errors'] > 0:
                        print(f"Errors: {stats['errors']}")

        elif args.command == 'summarize-pages':
            if not rag.chat_client:
                print("Error: No chat client configured.")
                print("Set OPENAI_API_KEY or configure RAG_CHAT_API_ENDPOINT")
                import sys
                sys.exit(1)

            from rag_system.ingestion.indexer import Indexer
            import json as json_module

            print("Generating page summaries...")
            indexer = Indexer(args.db, chat_client=rag.chat_client)
            stats = indexer.summarize_pages(page_id=args.page_id, force=args.force)

            if args.json:
                print(json_module.dumps(stats, indent=2))
            else:
                if stats.get('error'):
                    print(f"Error: {stats['error']}")
                else:
                    print("Page Summarization Results")
                    print("=" * 40)
                    print(f"Pages processed: {stats['pages_processed']}")
                    print(f"Summaries generated: {stats['summaries_generated']}")
                    if stats['errors'] > 0:
                        print(f"Errors: {stats['errors']}")

        elif args.command == 'rebuild-summaries':
            if not rag.chat_client:
                print("Error: No chat client configured.")
                print("Set OPENAI_API_KEY or configure RAG_CHAT_API_ENDPOINT")
                import sys
                sys.exit(1)

            from rag_system.summarization.summarizer import SummarizationPipeline
            import json as json_module

            print("Rebuilding system and global summaries...")
            pipeline = SummarizationPipeline(args.db, rag.chat_client)
            stats = pipeline.rebuild_all_summaries()

            if args.json:
                print(json_module.dumps(stats, indent=2))
            else:
                print("Summary Rebuild Results")
                print("=" * 40)
                print(f"Pages summarized: {stats['pages_summarized']}")
                print(f"Systems created: {stats['systems_created']}")
                print(f"Systems summarized: {stats['systems_summarized']}")
                print(f"Global summary: {'Yes' if stats['global_summary_generated'] else 'No'}")
                if stats['errors']:
                    print(f"Errors: {len(stats['errors'])}")
                    for error in stats['errors'][:5]:
                        print(f"  - {error}")

        elif args.command == 'backfill':
            if not rag.vector_client:
                print("Error: No vector client configured.")
                print("Set OPENAI_API_KEY or configure RAG_VECTOR_API_ENDPOINT")
                import sys
                sys.exit(1)

            backfill_embeddings(
                rag.db_path,
                rag.vector_client,
                batch_size=args.batch_size,
                force=args.force
            )

        elif args.command == 'crawl-sessions':
            from rag_system.database import list_crawl_sessions
            conn = get_connection(args.db)
            try:
                sessions = list_crawl_sessions(conn)
                if args.json:
                    import json
                    print(json.dumps(sessions, indent=2, default=str))
                else:
                    if not sessions:
                        print("No crawl sessions found.")
                    else:
                        print(f"{'ID':<6} {'Status':<12} {'URL':<50} {'Pages':<8} {'Started'}")
                        print("-" * 100)
                        for s in sessions:
                            started = s['started_at'][:19] if s['started_at'] else 'N/A'
                            print(f"{s['id']:<6} {s['status']:<12} {s['start_url'][:50]:<50} {s['pages_crawled']:<8} {started}")
            finally:
                conn.close()

        elif args.command == 'resume-embeddings':
            if not rag.vector_client:
                print("Error: No vector client configured.")
                print("Set OPENAI_API_KEY or configure RAG_VECTOR_API_ENDPOINT")
                import sys
                sys.exit(1)

            from rag_system.ingestion.indexer import Indexer

            indexer = Indexer(
                rag.db_path,
                vector_client=rag.vector_client,
                chat_client=rag.chat_client
            )

            page_id = getattr(args, 'page_id', None)
            batch_size = getattr(args, 'batch_size', config.EMBEDDING_BATCH_SIZE)

            if page_id:
                print(f"Resuming embedding generation for page {page_id}...")
            else:
                print("Resuming embedding generation for all chunks without embeddings...")

            try:
                result = indexer.resume_embeddings(
                    page_id=page_id,
                    batch_size=batch_size
                )

                if result.get('error'):
                    print(f"Error: {result['error']}")
                else:
                    print(f"\nEmbedding Resume Results:")
                    print(f"  Total chunks: {result['chunks_total']}")
                    print(f"  Embedded: {result['chunks_embedded']}")
                    print(f"  Skipped: {result['chunks_skipped']}")
                    print(f"  Failed: {result['chunks_failed']}")

                    if result.get('rate_limit_hit'):
                        print("\n  Note: Rate limit was hit during processing")

                    if result.get('completed'):
                        print("\n  Status: Completed")
                    else:
                        print("\n  Status: Interrupted - run again to continue")

            except RateLimitError as e:
                print(f"\nRate limit exceeded: {e}")
                print("Progress has been saved. Run the command again to resume.")
                import sys
                sys.exit(1)
            except KeyboardInterrupt:
                print("\n\nEmbedding interrupted. Progress has been saved - run the same command to resume.")
                logger.info("Embedding interrupted by user")

    except KeyboardInterrupt:
        print("\nShutdown requested")
        logger.info("Shutdown requested via keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        # Cleanup will be handled by atexit handlers
        logger.debug("Main function cleanup complete")


if __name__ == '__main__':
    main()
