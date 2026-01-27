"""Indexer module for the RAG system.

Orchestrates the full ingestion pipeline: crawling, parsing, chunking,
and storing in the database.
"""

from typing import Dict, List, Optional, Any
import os
import time

from rag_system import config
from rag_system.api_client import APIError
from rag_system.database import (
    get_connection, insert_page, get_page_by_url,
    insert_chunk, update_chunk_embedding,
    delete_chunks_by_page, delete_doc_terms_by_page, update_page_content,
    create_crawl_session, get_active_session_for_url, get_crawl_session,
    update_session_status, update_session_stats, acquire_session_lock,
    add_urls_to_crawl_queue, get_crawl_queue_urls, clear_crawl_queue
)
from rag_system.ingestion.crawler import Crawler
from rag_system.ingestion.parser import parse_html, extract_title
from rag_system.ingestion.chunker import chunk_markdown
from rag_system.ingestion.html_to_markdown import html_to_markdown
from rag_system.security import sanitize_html_content, validate_content_length
from rag_system.utils import hash_content, get_logger

logger = get_logger(__name__)


def build_contextual_text(chunk: Dict[str, Any], page_title: Optional[str] = None) -> str:
    """Build contextual text for embedding a chunk.

    Prepends page title and heading path to the chunk content so that
    embeddings capture broader context. This improves retrieval quality
    by 20-30% according to benchmarks (contextual retrieval / late chunking).

    Args:
        chunk: Chunk dict with 'content' and optionally 'heading_path'.
        page_title: Title of the page containing this chunk.

    Returns:
        Contextual text string ready for embedding.
    """
    parts = []

    if page_title:
        parts.append(f"Document: {page_title}")

    heading_path = chunk.get('heading_path', '')
    if heading_path:
        parts.append(f"Section: {heading_path}")

    if parts:
        parts.append('')

    parts.append(chunk['content'])
    return '\n'.join(parts)


class Indexer:
    """Orchestrates the document indexing pipeline."""

    def __init__(self, db_path: str, vector_client: Optional[Any] = None,
                 chat_client: Optional[Any] = None):
        """Initialize the indexer.

        Args:
            db_path: Path to the SQLite database.
            vector_client: Optional vector API client for embeddings.
            chat_client: Optional chat API client for entity extraction.
        """
        self.db_path = db_path
        self.vector_client = vector_client
        self.chat_client = chat_client

    def index_page(self, page_data: Dict[str, Any]) -> Optional[int]:
        """Index a single page.

        Args:
            page_data: Dict with 'url', 'html', and 'status_code'.

        Returns:
            Page ID if indexed, None if skipped.
        """
        url = page_data['url']
        html = page_data['html']

        # Validate content length
        is_valid, error = validate_content_length(html)
        if not is_valid:
            logger.warning(f"Content too large for {url}: {error}")
            return None

        # Sanitize HTML content to remove potentially dangerous elements
        html = sanitize_html_content(html)

        conn = get_connection(self.db_path)

        try:
            # Parse HTML for title and basic text
            parsed = parse_html(html, remove_nav=True, remove_footer=True)

            # Convert HTML to Markdown for chunking (Markdown-first approach)
            markdown = html_to_markdown(html)

            # Generate content hash
            content_hash = hash_content(html)
            title = parsed['title'] or extract_title(html)

            # Check if page already exists
            existing = get_page_by_url(conn, url)
            if existing:
                # Check if content changed
                if existing.get('content_hash') == content_hash:
                    logger.info(f"Skipping unchanged page: {url}")
                    return None

                # Content changed - perform incremental update
                page_id = existing['id']
                logger.info(f"Content changed for page: {url} - re-indexing")

                # Delete old data (terms must be deleted before chunks due to foreign key)
                delete_doc_terms_by_page(conn, page_id)
                delete_chunks_by_page(conn, page_id)

                # Update page content
                update_page_content(conn, page_id, title, html, markdown, content_hash)
            else:
                # New page - insert it
                page_id = insert_page(conn, url, title, html, markdown, content_hash)
                logger.info(f"Indexed new page: {url} (id={page_id})")

            # Create chunks from Markdown (heading structure is unambiguous in MD)
            chunk_result = chunk_markdown(
                markdown=markdown,
                small_chunk_size=config.SMALL_CHUNK_SIZE,
                large_chunk_size=config.LARGE_CHUNK_SIZE,
                overlap=config.CHUNK_OVERLAP
            )

            # Store large chunks
            large_chunk_ids = []
            for chunk in chunk_result['large_chunks']:
                chunk_id = insert_chunk(
                    conn,
                    page_id=page_id,
                    chunk_type='large',
                    chunk_index=chunk['index'],
                    content=chunk['content'],
                    heading_path=chunk['heading_path']
                )
                large_chunk_ids.append(chunk_id)

            # Store small chunks with parent references
            small_chunk_ids = []
            for chunk in chunk_result['small_chunks']:
                parent_id = None
                if chunk['parent_index'] is not None and chunk['parent_index'] < len(large_chunk_ids):
                    parent_id = large_chunk_ids[chunk['parent_index']]

                chunk_id = insert_chunk(
                    conn,
                    page_id=page_id,
                    chunk_type='small',
                    chunk_index=chunk['index'],
                    content=chunk['content'],
                    heading_path=chunk['heading_path'],
                    parent_chunk_id=parent_id
                )
                small_chunk_ids.append(chunk_id)

            logger.info(f"Created {len(large_chunk_ids)} large chunks, {len(small_chunk_ids)} small chunks")

            # Generate embeddings if vector client available
            # Uses contextual retrieval: embeds chunk with page title and heading path
            if self.vector_client and small_chunk_ids:
                page_title = parsed['title'] or extract_title(html)
                self._generate_embeddings(
                    conn, chunk_result['small_chunks'], small_chunk_ids, page_title
                )

            return page_id

        finally:
            conn.close()

    def _generate_embeddings(self, conn, chunks: List[Dict], chunk_ids: List[int],
                              page_title: Optional[str] = None) -> None:
        """Generate and store embeddings for chunks with batching and rate limiting.

        Embeds each chunk with its contextual information (page title, heading path)
        to improve retrieval quality. Processes in batches with retry logic for
        rate limit errors.

        Args:
            conn: Database connection.
            chunks: List of chunk dicts with 'content' and 'heading_path'.
            chunk_ids: List of chunk IDs in database.
            page_title: Optional page title for contextual embedding.
        """
        if not chunks:
            return

        batch_size = config.EMBEDDING_BATCH_SIZE
        batch_delay = config.EMBEDDING_BATCH_DELAY
        max_retries = config.EMBEDDING_MAX_RETRIES
        total = len(chunks)
        embedded_count = 0

        # Process chunks in batches
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_ids = chunk_ids[i:i + batch_size]
            texts = [self._build_contextual_text(c, page_title) for c in batch_chunks]

            # Retry with exponential backoff on rate limit errors
            for attempt in range(max_retries):
                try:
                    embeddings = self.vector_client.get_embeddings_batch(texts)
                    for chunk_id, embedding in zip(batch_ids, embeddings):
                        update_chunk_embedding(conn, chunk_id, embedding)
                    embedded_count += len(embeddings)
                    break

                except APIError as e:
                    is_rate_limit = e.status_code == 429
                    can_retry = attempt < max_retries - 1

                    if is_rate_limit and can_retry:
                        delay = 2 ** attempt
                        logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        logger.warning(f"Failed to embed batch starting at {i}: {e}")
                        break

                except Exception as e:
                    logger.warning(f"Failed to embed batch starting at {i}: {e}")
                    break

            # Progress logging for large jobs
            if total > batch_size:
                logger.info(f"Embedded {embedded_count}/{total} chunks")

            # Rate limit delay between batches
            if i + batch_size < total and batch_delay > 0:
                time.sleep(batch_delay)

        logger.info(f"Generated {embedded_count} embeddings")

    def _build_contextual_text(self, chunk: Dict[str, Any],
                               page_title: Optional[str] = None) -> str:
        """Build contextual text for embedding a chunk.

        Prepends page title and heading path to the chunk content so that
        embeddings capture broader context.

        Args:
            chunk: Chunk dict with 'content' and optionally 'heading_path'.
            page_title: Title of the page containing this chunk.

        Returns:
            Contextual text string ready for embedding.
        """
        parts = []

        if page_title:
            parts.append(f"Document: {page_title}")

        heading_path = chunk.get('heading_path', '')
        if heading_path:
            parts.append(f"Section: {heading_path}")

        if parts:
            parts.append('')

        parts.append(chunk['content'])
        return '\n'.join(parts)

    def index_pages(self, pages: List[Dict[str, Any]]) -> List[Optional[int]]:
        """Index multiple pages.

        Args:
            pages: List of page data dicts.

        Returns:
            List of page IDs (None for skipped pages).
        """
        results = []
        for page_data in pages:
            try:
                page_id = self.index_page(page_data)
                results.append(page_id)
            except Exception as e:
                logger.error(f"Failed to index {page_data.get('url')}: {e}")
                results.append(None)
        return results

    def crawl_and_index(self, start_url: str, allowed_domains: List[str],
                        excluded_paths: Optional[List[str]] = None,
                        included_paths: Optional[List[str]] = None,
                        max_pages: int = 1000,
                        delay: float = 1.0,
                        ignore_robots: bool = False,
                        fresh: bool = False,
                        basic_auth_token: Optional[str] = None) -> Dict[str, int]:
        """Crawl a website and index all pages.

        Supports resuming interrupted crawls. If a previous crawl for the same
        URL was interrupted, it will automatically resume from where it left off
        unless fresh=True is specified.

        Args:
            start_url: Starting URL.
            allowed_domains: List of allowed domains.
            excluded_paths: List of path prefixes to exclude.
            included_paths: List of path prefixes to include. If set, only URLs
                          whose path starts with one of these will be crawled.
            max_pages: Maximum pages to crawl.
            delay: Delay between requests.
            ignore_robots: If True, ignore robots.txt restrictions.
            fresh: If True, start a new crawl even if a resumable session exists.
            basic_auth_token: Optional Base64-encoded token for HTTP Basic Auth.

        Returns:
            Dict with crawl/index statistics.
        """
        conn = get_connection(self.db_path)
        session_id = None
        resuming = False

        try:
            # Check for existing resumable session
            if not fresh:
                existing_session = get_active_session_for_url(conn, start_url)
                if existing_session and existing_session['status'] == 'interrupted':
                    # Try to acquire lock on the session
                    if acquire_session_lock(conn, existing_session['id']):
                        session_id = existing_session['id']
                        resuming = True
                        logger.info(f"Resuming interrupted session {session_id}")

            # Create new session if not resuming
            if session_id is None:
                session_id = create_crawl_session(
                    conn, start_url, allowed_domains, max_pages
                )
                logger.info(f"Created new crawl session {session_id}")

            # Create crawler
            crawler = Crawler(
                start_url=start_url,
                allowed_domains=allowed_domains,
                excluded_paths=excluded_paths or config.EXCLUDED_PATHS,
                included_paths=included_paths or config.INCLUDED_PATHS,
                max_pages=max_pages,
                delay=delay,
                cache_dir=os.environ.get('RAG_HTTP_CACHE_DIR'),
                ignore_robots=ignore_robots,
                basic_auth_token=basic_auth_token
            )

            # If resuming, load queue from database
            if resuming:
                queue_urls = get_crawl_queue_urls(conn, session_id)
                if queue_urls:
                    crawler.queue = list(queue_urls)
                    logger.info(f"Loaded {len(queue_urls)} URLs from saved queue")
                # Clear the queue in DB since we've loaded it into memory
                clear_crawl_queue(conn, session_id)

            pages_crawled = 0
            pages_indexed = 0
            pages_skipped = 0
            errors = 0

            try:
                for page_data in crawler.crawl():
                    pages_crawled += 1
                    try:
                        page_id = self.index_page(page_data)
                        if page_id:
                            pages_indexed += 1
                        else:
                            pages_skipped += 1
                    except Exception as e:
                        logger.error(f"Error indexing {page_data.get('url')}: {e}")
                        errors += 1

                    # Update session stats periodically
                    if pages_crawled % 10 == 0:
                        update_session_stats(
                            conn, session_id,
                            pages_crawled=pages_crawled,
                            pages_indexed=pages_indexed,
                            pages_skipped=pages_skipped,
                            errors=errors
                        )

                # Crawl completed successfully
                update_session_stats(
                    conn, session_id,
                    pages_crawled=pages_crawled,
                    pages_indexed=pages_indexed,
                    pages_skipped=pages_skipped,
                    errors=errors
                )
                update_session_status(conn, session_id, 'completed')
                logger.info(f"Crawl session {session_id} completed")

            except KeyboardInterrupt:
                # Save queue state for resume
                logger.info("Crawl interrupted, saving state...")
                if crawler.queue:
                    add_urls_to_crawl_queue(conn, session_id, crawler.queue)
                    logger.info(f"Saved {len(crawler.queue)} URLs to queue")
                update_session_stats(
                    conn, session_id,
                    pages_crawled=pages_crawled,
                    pages_indexed=pages_indexed,
                    pages_skipped=pages_skipped,
                    errors=errors
                )
                update_session_status(conn, session_id, 'interrupted')
                logger.info(f"Crawl session {session_id} interrupted")
                raise  # Re-raise to let caller handle

            # Build BM25 index for search
            from rag_system.search.bm25_search import BM25Index
            logger.info("Building BM25 search index...")
            bm25_index = BM25Index(self.db_path)
            bm25_index.build()
            logger.info("BM25 index built successfully")

            return {
                'pages_crawled': pages_crawled,
                'pages_indexed': pages_indexed,
                'pages_skipped': pages_skipped,
                'errors': errors,
                'session_id': session_id,
                'resumed': resuming
            }

        finally:
            conn.close()

    def get_stats(self) -> Dict[str, int]:
        """Get index statistics.

        Returns:
            Dict with total_pages, total_chunks, etc.
        """
        conn = get_connection(self.db_path)

        try:
            cursor = conn.cursor()

            # Count pages
            cursor.execute("SELECT COUNT(*) FROM pages")
            total_pages = cursor.fetchone()[0]

            # Count chunks
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]

            # Count chunks with embeddings
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE embedding_json IS NOT NULL")
            chunks_with_embeddings = cursor.fetchone()[0]

            # Count entities
            cursor.execute("SELECT COUNT(*) FROM entities")
            total_entities = cursor.fetchone()[0]

            return {
                'total_pages': total_pages,
                'total_chunks': total_chunks,
                'chunks_with_embeddings': chunks_with_embeddings,
                'total_entities': total_entities
            }

        finally:
            conn.close()
