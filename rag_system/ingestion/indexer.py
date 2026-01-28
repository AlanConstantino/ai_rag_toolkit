"""Indexer module for the RAG system.

Orchestrates the full ingestion pipeline: crawling, parsing, chunking,
and storing in the database.
"""

from typing import Dict, List, Optional, Any
import os
import time

from rag_system import config
from rag_system.api_client import APIError, RateLimitError
from rag_system.database import (
    get_connection, insert_page, get_page_by_url,
    insert_chunk, update_chunk_embedding,
    delete_chunks_by_page, delete_doc_terms_by_page, update_page_content,
    create_crawl_session, get_active_session_for_url, get_crawl_session,
    update_session_status, update_session_stats, acquire_session_lock,
    add_urls_to_crawl_queue, get_crawl_queue_urls, clear_crawl_queue,
    create_embedding_job, update_embedding_job_progress, update_embedding_job_status,
    get_chunks_without_embeddings,
    insert_entity, insert_relationship, link_chunk_to_entity, get_entity_by_name,
    get_all_pages
)
from rag_system.shutdown import is_shutdown_requested
from rag_system.ingestion.crawler import Crawler
from rag_system.ingestion.parser import parse_html, extract_title
from rag_system.ingestion.chunker import chunk_markdown
from rag_system.ingestion.html_to_markdown import html_to_markdown
from rag_system.security import sanitize_html_content, validate_content_length
from rag_system.utils import hash_content, get_logger

logger = get_logger(__name__)


class EmbeddingStats:
    """Statistics tracker for embedding operations.

    Tracks progress and outcomes during embedding generation, similar to CrawlStats.
    """

    def __init__(self):
        """Initialize embedding statistics."""
        self.chunks_total: int = 0
        self.chunks_embedded: int = 0
        self.chunks_skipped: int = 0
        self.chunks_failed: int = 0
        self.rate_limit_hit: bool = False
        self.interrupted: bool = False
        self.error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary.

        Returns:
            Dict with all statistics.
        """
        return {
            'chunks_total': self.chunks_total,
            'chunks_embedded': self.chunks_embedded,
            'chunks_skipped': self.chunks_skipped,
            'chunks_failed': self.chunks_failed,
            'rate_limit_hit': self.rate_limit_hit,
            'interrupted': self.interrupted,
            'error_message': self.error_message
        }


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
                    logger.debug(f"Skipping unchanged page: {url}")
                    return None

                # Content changed - perform incremental update
                page_id = existing['id']
                logger.debug(f"Content changed for page: {url} - re-indexing")

                # Delete old data (terms must be deleted before chunks due to foreign key)
                delete_doc_terms_by_page(conn, page_id)
                delete_chunks_by_page(conn, page_id)

                # Update page content
                update_page_content(conn, page_id, title, html, markdown, content_hash)
            else:
                # New page - insert it
                page_id = insert_page(conn, url, title, html, markdown, content_hash)
                logger.debug(f"Indexed new page: {url} (id={page_id})")

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

            logger.debug(f"Created {len(large_chunk_ids)} large chunks, {len(small_chunk_ids)} small chunks")

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
                              page_title: Optional[str] = None) -> EmbeddingStats:
        """Generate and store embeddings for chunks with batching and rate limiting.

        Embeds each chunk with its contextual information (page title, heading path)
        to improve retrieval quality. Processes in batches with retry logic for
        rate limit errors.

        Supports graceful interruption via KeyboardInterrupt or shutdown signals.
        Progress is saved to allow resumption.

        Args:
            conn: Database connection.
            chunks: List of chunk dicts with 'content' and 'heading_path'.
            chunk_ids: List of chunk IDs in database.
            page_title: Optional page title for contextual embedding.

        Returns:
            EmbeddingStats with progress information.

        Raises:
            RateLimitError: If rate limit is hit and EMBEDDING_STOP_ON_RATE_LIMIT is True.
            KeyboardInterrupt: If interrupted (progress is saved first).
        """
        stats = EmbeddingStats()

        if not chunks:
            return stats

        batch_size = config.EMBEDDING_BATCH_SIZE
        batch_delay = config.EMBEDDING_BATCH_DELAY
        max_retries = config.EMBEDDING_MAX_RETRIES
        retry_delay = config.EMBEDDING_RETRY_DELAY
        stop_on_rate_limit = config.EMBEDDING_STOP_ON_RATE_LIMIT

        stats.chunks_total = len(chunks)

        try:
            # Process chunks in batches
            for i in range(0, stats.chunks_total, batch_size):
                # Check for shutdown request between batches
                if is_shutdown_requested():
                    logger.info("Shutdown requested, saving embedding progress...")
                    stats.interrupted = True
                    break

                batch_chunks = chunks[i:i + batch_size]
                batch_ids = chunk_ids[i:i + batch_size]
                texts = [self._build_contextual_text(c, page_title) for c in batch_chunks]

                batch_success = False
                # Retry with exponential backoff on rate limit errors
                for attempt in range(max_retries):
                    try:
                        embeddings = self.vector_client.get_embeddings_batch(texts)
                        for chunk_id, embedding in zip(batch_ids, embeddings):
                            update_chunk_embedding(conn, chunk_id, embedding)
                        stats.chunks_embedded += len(embeddings)
                        batch_success = True
                        break

                    except APIError as e:
                        is_rate_limit = e.status_code == 429
                        can_retry = attempt < max_retries - 1

                        if is_rate_limit:
                            stats.rate_limit_hit = True
                            # Extract retry_after if available
                            retry_after = retry_delay * (2 ** attempt)

                            if stop_on_rate_limit and not can_retry:
                                # Raise RateLimitError to stop embedding
                                stats.error_message = f"Rate limit exceeded after {max_retries} retries"
                                raise RateLimitError(
                                    message=stats.error_message,
                                    retry_after=retry_after,
                                    status_code=429
                                )

                            if can_retry:
                                logger.warning(f"Rate limited, retrying in {retry_after:.1f}s (attempt {attempt + 1}/{max_retries})")
                                time.sleep(retry_after)
                            else:
                                logger.warning(f"Failed to embed batch starting at {i} after {max_retries} retries: {e}")
                                stats.chunks_failed += len(batch_ids)
                                break
                        else:
                            if can_retry:
                                delay = retry_delay * (2 ** attempt)
                                logger.warning(f"API error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}")
                                time.sleep(delay)
                            else:
                                logger.warning(f"Failed to embed batch starting at {i}: {e}")
                                stats.chunks_failed += len(batch_ids)
                                break

                    except Exception as e:
                        logger.warning(f"Failed to embed batch starting at {i}: {e}")
                        stats.chunks_failed += len(batch_ids)
                        break

                # Progress logging for large jobs
                if stats.chunks_total > batch_size:
                    logger.info(f"Embedded {stats.chunks_embedded}/{stats.chunks_total} chunks")

                # Rate limit delay between batches
                if i + batch_size < stats.chunks_total and batch_delay > 0:
                    time.sleep(batch_delay)

        except KeyboardInterrupt:
            # Save progress before re-raising
            logger.info("Embedding interrupted, progress saved")
            stats.interrupted = True
            raise

        logger.info(f"Generated {stats.chunks_embedded} embeddings")
        return stats

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

                    # Log progress at INFO level every 10 pages
                    if pages_crawled % 10 == 0:
                        logger.info(f"Index progress: {pages_indexed} indexed, {pages_skipped} skipped, {errors} errors")

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

    def resume_embeddings(self, page_id: Optional[int] = None,
                           batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Resume embedding generation for chunks that don't have embeddings.

        Queries chunks with embedding_json IS NULL and generates embeddings
        for them. Supports interruption and can be resumed again if stopped.

        Args:
            page_id: Optional page ID to filter chunks. If None, processes all
                     chunks without embeddings.
            batch_size: Number of chunks to embed per API call.

        Returns:
            Dict with statistics about the resume operation.

        Raises:
            RateLimitError: If rate limit is hit and EMBEDDING_STOP_ON_RATE_LIMIT is True.
        """
        if not self.vector_client:
            logger.warning("No vector client configured, cannot generate embeddings")
            return {'error': 'No vector client configured'}

        batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        conn = get_connection(self.db_path)

        try:
            # Find chunks without embeddings
            chunks_to_embed = get_chunks_without_embeddings(conn, page_id)
            total = len(chunks_to_embed)

            if total == 0:
                logger.info("No chunks found without embeddings")
                return {
                    'chunks_total': 0,
                    'chunks_embedded': 0,
                    'chunks_skipped': 0,
                    'chunks_failed': 0,
                    'completed': True
                }

            logger.info(f"Found {total} chunks without embeddings")

            # Create embedding job for tracking
            job_id = create_embedding_job(conn, page_id, total)
            logger.info(f"Created embedding job {job_id}")

            # Build chunk dicts and ids for _generate_embeddings
            chunks = []
            chunk_ids = []
            page_titles = {}

            for row in chunks_to_embed:
                chunks.append({
                    'content': row['content'],
                    'heading_path': row['heading_path']
                })
                chunk_ids.append(row['id'])
                # Cache page titles
                if row['title'] not in page_titles:
                    page_titles[row['id']] = row['title']

            # For simplicity, use the first page title (for contextual embedding)
            # In a more sophisticated implementation, we'd group by page
            first_title = chunks_to_embed[0]['title'] if chunks_to_embed else None

            try:
                stats = self._generate_embeddings(conn, chunks, chunk_ids, first_title)

                # Update job progress
                update_embedding_job_progress(
                    conn, job_id,
                    chunks_embedded=stats.chunks_embedded,
                    chunks_skipped=stats.chunks_skipped,
                    chunks_failed=stats.chunks_failed
                )

                if stats.interrupted:
                    update_embedding_job_status(conn, job_id, 'interrupted')
                elif stats.chunks_failed > 0:
                    update_embedding_job_status(
                        conn, job_id, 'completed',
                        error_message=f"{stats.chunks_failed} chunks failed"
                    )
                else:
                    update_embedding_job_status(conn, job_id, 'completed')

                result = stats.to_dict()
                result['job_id'] = job_id
                result['completed'] = not stats.interrupted
                return result

            except RateLimitError as e:
                # Update job status with rate limit error
                update_embedding_job_status(
                    conn, job_id, 'interrupted',
                    error_message=str(e)
                )
                raise

            except KeyboardInterrupt:
                # Update job status on interrupt
                update_embedding_job_status(conn, job_id, 'interrupted')
                raise

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

    def extract_entities(self, page_id: Optional[int] = None) -> Dict[str, Any]:
        """Extract entities from pages as a post-processing step.

        This is a standalone operation that runs after crawling is complete.
        It extracts entities from page text and stores them in the database.

        Args:
            page_id: Optional specific page ID to process. If None, processes
                     all pages that haven't had entities extracted yet.

        Returns:
            Dict with statistics about the extraction operation.
        """
        if not self.chat_client:
            logger.warning("No chat client configured for entity extraction")
            return {'error': 'No chat client configured'}

        from rag_system.knowledge_graph.entity_extractor import EntityExtractor

        stats = {
            'pages_processed': 0,
            'entities_extracted': 0,
            'relationships_extracted': 0,
            'errors': 0
        }

        conn = get_connection(self.db_path)
        try:
            # Get pages to process
            if page_id:
                cursor = conn.execute(
                    "SELECT id, parsed_text FROM pages WHERE id = ?",
                    (page_id,)
                )
            else:
                # Get pages that don't have entities yet
                cursor = conn.execute("""
                    SELECT p.id, p.parsed_text FROM pages p
                    WHERE NOT EXISTS (
                        SELECT 1 FROM chunk_entities ce
                        JOIN chunks c ON ce.chunk_id = c.id
                        WHERE c.page_id = p.id
                    )
                """)

            pages = cursor.fetchall()
            total_pages = len(pages)

            if total_pages == 0:
                logger.info("No pages found needing entity extraction")
                return stats

            logger.info(f"Extracting entities from {total_pages} pages...")
            extractor = EntityExtractor(self.chat_client)

            for page in pages:
                page_id = page['id']
                text = page['parsed_text'] or ''

                if not text.strip():
                    continue

                try:
                    # Extract entities
                    result = extractor.extract(text)
                    entities = result.get('entities', [])
                    relationships = result.get('relationships', [])

                    # Get chunk IDs for this page
                    chunk_cursor = conn.execute(
                        "SELECT id FROM chunks WHERE page_id = ?",
                        (page_id,)
                    )
                    chunk_ids = [row['id'] for row in chunk_cursor.fetchall()]

                    # Store entities
                    entity_ids = {}
                    for entity in entities:
                        name = entity.get('name', '')
                        entity_type = entity.get('type', 'concept')
                        description = entity.get('description', '')

                        if not name:
                            continue

                        existing = get_entity_by_name(conn, name)
                        if existing:
                            entity_id = existing['id']
                        else:
                            entity_id = insert_entity(conn, name, entity_type, description)
                            stats['entities_extracted'] += 1

                        entity_ids[name] = entity_id

                        # Link entity to chunks
                        for chunk_id in chunk_ids:
                            link_chunk_to_entity(conn, chunk_id, entity_id)

                    # Store relationships
                    for rel in relationships:
                        source_name = rel.get('source', '')
                        target_name = rel.get('target', '')
                        rel_type = rel.get('type', 'related_to')
                        description = rel.get('description', '')

                        source_id = entity_ids.get(source_name)
                        target_id = entity_ids.get(target_name)

                        if source_id and target_id:
                            insert_relationship(conn, source_id, target_id, rel_type, description)
                            stats['relationships_extracted'] += 1

                    stats['pages_processed'] += 1
                    logger.debug(f"Extracted {len(entities)} entities from page {page_id}")

                except Exception as e:
                    logger.warning(f"Entity extraction failed for page {page_id}: {e}")
                    stats['errors'] += 1

                # Progress logging
                if stats['pages_processed'] % 10 == 0:
                    logger.info(f"Progress: {stats['pages_processed']}/{total_pages} pages")

            return stats

        finally:
            conn.close()
