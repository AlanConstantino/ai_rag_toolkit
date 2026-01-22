"""Main CLI module for the RAG system.

Provides command-line interface for interacting with the RAG system.
"""

import argparse
import hashlib
import os
from typing import Dict, List, Any, Tuple, Optional

from rag_system import config
from rag_system.database import (
    init_db, get_connection, cache_query, get_cached_query, log_query
)
from rag_system.api_client import (
    VectorAPIClient, ChatAPIClient,
    create_openai_vector_client, create_openai_chat_client
)
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
from rag_system.utils import get_logger

logger = get_logger(__name__)


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
        '--ignore-robots', action='store_true',
        help='Ignore robots.txt restrictions (use responsibly)'
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
        if vector_client is None and chat_client is None:
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                logger.info("Using OpenAI API for embeddings and chat")
                vector_client = create_openai_vector_client()
                chat_client = create_openai_chat_client()

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

    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """Query the system.

        Args:
            question: Question to ask.
            top_k: Number of results.

        Returns:
            Result dict with answer, chunks, confidence.
        """
        top_k = top_k or config.TOP_K_FINAL

        # Generate query hash for caching
        query_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()

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
            logger.info(f"Using cached query processing for: {question[:50]}...")
        else:
            # Classify query
            query_type = self.classifier.classify(question)
            # Expand query
            expanded = self.expander.expand(question)

        # Search - use BM25 only if no vector client, otherwise hybrid
        all_results = []
        for q in expanded:
            if self.vector_client:
                # Get embedding and do hybrid search
                try:
                    query_embedding = self.vector_client.get_embedding(q)
                    results = self.hybrid_search.search(query_embedding, q)
                except Exception as e:
                    logger.warning(f"Vector search failed, falling back to BM25: {e}")
                    results = self.bm25_search.search(q)
            else:
                # BM25 only
                results = self.bm25_search.search(q)
            all_results.extend(results)

        # Deduplicate by chunk_id
        seen = set()
        unique_results = []
        for chunk_id, score in all_results:
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_results.append((chunk_id, score))

        # Rerank
        reranked = self.reranker.rerank(unique_results, question)

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
        if chunks and self.chat_client:
            context = self.context_builder.build_context(chunks)
            answer_result = self.answer_generator.generate_with_confidence(
                question, context, metrics['overall'], query_type
            )
            answer = answer_result['answer']
            if answer_result.get('disclaimer'):
                answer = f"{answer}\n\n{answer_result['disclaimer']}"
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

        # Log the query
        conn = get_connection(self.db_path)
        try:
            chunk_ids = [c['id'] for c in chunks]
            log_query(
                conn, question, query_type, expanded, chunk_ids,
                metrics['overall'], bool(chunks and self.chat_client)
            )
        finally:
            conn.close()

        return {
            'query': question,
            'query_type': query_type,
            'answer': answer,
            'chunks': chunks,
            'confidence': metrics['overall'],
            'metrics': metrics
        }

    def ingest(self, start_url: str, max_pages: int = None,
                ignore_robots: bool = False) -> Dict[str, Any]:
        """Ingest documentation from URL.

        Args:
            start_url: Starting URL to crawl.
            max_pages: Maximum pages to crawl.
            ignore_robots: If True, ignore robots.txt restrictions.

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
            ignore_robots=ignore_robots
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
            print("  quit/exit         - Exit")
        elif cmd == 'stats':
            stats = rag.get_stats()
            print(format_stats(stats))
        elif cmd == 'query' and args:
            result = rag.query(args[0])
            print(format_query_result(result))
        else:
            # Treat as query
            result = rag.query(user_input)
            print(format_query_result(result))

        print()


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    rag = RAGSystem(db_path=args.db)

    if args.command == 'ingest':
        print(f"Ingesting from {args.url}...")
        stats = rag.ingest(
            args.url,
            max_pages=args.max_pages,
            ignore_robots=args.ignore_robots
        )
        print(f"Crawled {stats['pages_crawled']} pages, indexed {stats['pages_indexed']}, skipped {stats['pages_skipped']}, errors: {stats['errors']}")

    elif args.command == 'query':
        result = rag.query(args.question, top_k=args.top_k)
        print(format_query_result(result))

    elif args.command == 'stats':
        stats = rag.get_stats()
        print(format_stats(stats))

    elif args.command == 'interactive':
        run_interactive(rag)


if __name__ == '__main__':
    main()
