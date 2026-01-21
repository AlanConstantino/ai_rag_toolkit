"""Indexer module for the RAG system.

Orchestrates the full ingestion pipeline: crawling, parsing, chunking,
and storing in the database.
"""

from typing import Dict, List, Optional, Any, Generator
import os

from rag_system import config
from rag_system.database import (
    get_connection, insert_page, get_page_by_url,
    insert_chunk, update_chunk_embedding
)
from rag_system.ingestion.crawler import Crawler
from rag_system.ingestion.parser import parse_html, extract_title
from rag_system.ingestion.chunker import chunk_markdown
from rag_system.ingestion.html_to_markdown import html_to_markdown
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

        conn = get_connection(self.db_path)

        try:
            # Check if page already exists
            existing = get_page_by_url(conn, url)
            if existing:
                # Check if content changed
                content_hash = hash_content(html)
                if existing.get('content_hash') == content_hash:
                    logger.info(f"Skipping unchanged page: {url}")
                    return None
                # TODO: Handle page updates
                logger.info(f"Page exists, skipping: {url}")
                return existing['id']

            # Parse HTML for title and basic text
            parsed = parse_html(html, remove_nav=True, remove_footer=True)

            # Convert HTML to Markdown for chunking (Markdown-first approach)
            # This naturally filters out nav/sidebar/footer as they don't convert
            # to meaningful Markdown structure
            markdown = html_to_markdown(html)

            # Generate content hash
            content_hash = hash_content(html)

            # Insert page
            page_id = insert_page(
                conn,
                url=url,
                title=parsed['title'] or extract_title(html),
                raw_html=html,
                parsed_text=markdown,  # Store markdown instead of parsed text
                content_hash=content_hash
            )

            logger.info(f"Indexed page: {url} (id={page_id})")

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
        """Generate and store embeddings for chunks using contextual retrieval.

        Embeds each chunk with its contextual information (page title, heading path)
        to improve retrieval quality. The embedding captures the broader context,
        but only the original chunk content is stored in the database.

        Args:
            conn: Database connection.
            chunks: List of chunk dicts with 'content' and 'heading_path'.
            chunk_ids: List of chunk IDs in database.
            page_title: Title of the page for contextual embedding.
        """
        try:
            texts = [build_contextual_text(c, page_title) for c in chunks]
            embeddings = self.vector_client.get_embeddings_batch(texts)

            for chunk_id, embedding in zip(chunk_ids, embeddings):
                update_chunk_embedding(conn, chunk_id, embedding)

            logger.info(f"Generated {len(embeddings)} embeddings with contextual retrieval")

        except Exception as e:
            logger.warning(f"Failed to generate embeddings: {e}")

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
                        ignore_robots: bool = False) -> Dict[str, int]:
        """Crawl a website and index all pages.

        Args:
            start_url: Starting URL.
            allowed_domains: List of allowed domains.
            excluded_paths: List of path prefixes to exclude.
            included_paths: List of path prefixes to include. If set, only URLs
                          whose path starts with one of these will be crawled.
            max_pages: Maximum pages to crawl.
            delay: Delay between requests.
            ignore_robots: If True, ignore robots.txt restrictions.

        Returns:
            Dict with crawl/index statistics.
        """
        crawler = Crawler(
            start_url=start_url,
            allowed_domains=allowed_domains,
            excluded_paths=excluded_paths or config.EXCLUDED_PATHS,
            included_paths=included_paths or config.INCLUDED_PATHS,
            max_pages=max_pages,
            delay=delay,
            cache_dir=os.environ.get('RAG_HTTP_CACHE_DIR'),
            ignore_robots=ignore_robots
        )

        pages_crawled = 0
        pages_indexed = 0
        pages_skipped = 0
        errors = 0

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
            'errors': errors
        }

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
