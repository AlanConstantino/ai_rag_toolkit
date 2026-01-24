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
    create_openai_vector_client, create_openai_chat_client
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

    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument(
        '--top-k', type=int, default=config.TOP_K_FINAL,
        help='Number of results to return'
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

    # Crawl sessions command
    sessions_parser = subparsers.add_parser(
        'crawl-sessions',
        help='View and manage crawl sessions'
    )
    sessions_parser.add_argument(
        '--json', action='store_true',
        help='Output as JSON'
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

    # Sources
    chunks = result.get('chunks', [])
    if chunks:
        lines.append("")
        lines.append("Sources:")
        for i, chunk in enumerate(chunks[:3], 1):
            score = chunk.get('score', 0)
            content = chunk.get('content', '')[:100]
            lines.append(f"  {i}. [{score:.2f}] {content}...")

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

        # Generate query hash for caching
        query_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()

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
        embedding_start = time.time()
        search_start = time.time()

        for q in expanded:
            # BM25 is always the foundation
            # Vector search enhances when AI is enabled and vector_client available
            query_embedding = None

            if config.AI_ENABLED and self.vector_client:
                try:
                    embed_start = time.time()
                    query_embedding = self.vector_client.get_embedding(q)
                    timing_metrics['embedding_time'] = timing_metrics.get('embedding_time', 0) + (time.time() - embed_start)
                except Exception as e:
                    logger.warning(f"Vector embedding failed, using BM25 only: {e}")

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

        # Build chunk info
        chunks = []
        conn = get_connection(self.db_path)
        try:
            for chunk_id, score in diversified:
                cursor = conn.execute(
                    "SELECT content, heading_path FROM chunks WHERE id = ?",
                    (chunk_id,)
                )
                row = cursor.fetchone()
                if row:
                    chunks.append({
                        'id': chunk_id,
                        'score': score,
                        'content': row['content'],
                        'heading_path': row['heading_path']
                    })
        finally:
            conn.close()

        # Calculate confidence
        metrics = self.confidence_analyzer.analyze(question, chunks)

        # Build context and generate answer
        generation_start = time.time()
        if chunks and self.chat_client:
            context = self.context_builder.build_context(chunks)
            answer_result = self.answer_generator.generate_with_confidence(
                question, context, metrics['overall'], query_type
            )
            answer = answer_result['answer']
            if answer_result.get('disclaimer'):
                answer = f"{answer}\n\n{answer_result['disclaimer']}"
            timing_metrics['generation_time'] = time.time() - generation_start
        else:
            answer = "No relevant information found." if not chunks else \
                     "Answer generation is not configured."

        # Cache query processing if not already cached
        if not cached:
            conn = get_connection(self.db_path)
            try:
                # Get embedding for caching (use first expanded query)
                query_embedding = []
                if self.vector_client and expanded:
                    try:
                        query_embedding = self.vector_client.get_embedding(expanded[0])
                    except Exception:
                        pass
                cache_query(conn, query_hash, query_type, expanded, query_embedding)
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

        result = {
            'query': question,
            'query_type': query_type,
            'answer': answer,
            'chunks': chunks,
            'confidence': metrics['overall'],
            'metrics': metrics,
            'timing': timing_metrics,
            'cache_hit': False
        }

        # Store result in cache
        if use_cache and _query_cache is not None:
            _query_cache.put(cache_key, result)
            logger.debug(f"Cached result for query: {sanitize_for_logging(question, 50)}")

        return result

    def ingest(self, start_url: str, max_pages: int = None,
                ignore_robots: bool = False, fresh: bool = False) -> Dict[str, Any]:
        """Ingest documentation from URL.

        Supports resuming interrupted crawls. If a previous crawl for the same
        URL was interrupted, it will automatically resume from where it left off
        unless fresh=True is specified.

        Args:
            start_url: Starting URL to crawl.
            max_pages: Maximum pages to crawl.
            ignore_robots: If True, ignore robots.txt restrictions.
            fresh: If True, start a new crawl even if a resumable session exists.

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
            fresh=fresh
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
                         batch_size: int = 100) -> Dict[str, int]:
    """Generate embeddings for chunks that don't have them.

    Args:
        db_path: Path to SQLite database.
        vector_client: Vector API client for embeddings.
        batch_size: Number of chunks to embed per API call.

    Returns:
        Dict with statistics about the backfill operation.
    """
    import time
    from rag_system.database import get_connection, update_chunk_embedding
    from rag_system.ingestion.indexer import build_contextual_text
    from rag_system.api_client import APIError

    conn = get_connection(db_path)
    try:
        # Find chunks without embeddings
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
            print("All chunks already have embeddings!")
            return {'total': 0, 'embedded': 0, 'errors': 0}

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
            try:
                stats = rag.ingest(
                    args.url,
                    max_pages=max_pages,
                    ignore_robots=args.ignore_robots,
                    fresh=args.fresh
                )
                resumed_msg = " (resumed)" if stats.get('resumed') else ""
                print(f"Crawled {stats['pages_crawled']} pages{resumed_msg}, indexed {stats['pages_indexed']}, skipped {stats['pages_skipped']}, errors: {stats['errors']}")
            except KeyboardInterrupt:
                print("\nIngestion interrupted. Progress has been saved - run the same command to resume.")
                logger.info("Ingestion interrupted by user")

        elif args.command == 'query':
            result = rag.query(args.question, top_k=args.top_k)
            print(format_query_result(result))

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

        elif args.command == 'backfill':
            if not rag.vector_client:
                print("Error: No vector client configured.")
                print("Set OPENAI_API_KEY or configure RAG_VECTOR_API_ENDPOINT")
                import sys
                sys.exit(1)

            backfill_embeddings(
                rag.db_path,
                rag.vector_client,
                batch_size=args.batch_size
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
