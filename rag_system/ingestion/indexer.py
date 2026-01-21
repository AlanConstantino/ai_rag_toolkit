"""Indexer module for the RAG system.

Orchestrates the full ingestion pipeline: crawling, parsing, chunking,
and storing in the database.
"""

from typing import Dict, List, Optional, Any, Generator

from rag_system import config
from rag_system.database import (
    get_connection, insert_page, get_page_by_url,
    insert_chunk, update_chunk_embedding
)
from rag_system.ingestion.crawler import Crawler
from rag_system.ingestion.parser import parse_html, extract_title
from rag_system.ingestion.chunker import chunk_document
from rag_system.utils import hash_content, get_logger

logger = get_logger(__name__)


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

            # Parse HTML
            parsed = parse_html(html, remove_nav=True, remove_footer=True)

            # Generate content hash
            content_hash = hash_content(html)

            # Insert page
            page_id = insert_page(
                conn,
                url=url,
                title=parsed['title'] or extract_title(html),
                raw_html=html,
                parsed_text=parsed['text'],
                content_hash=content_hash
            )

            logger.info(f"Indexed page: {url} (id={page_id})")

            # Create chunks
            chunk_result = chunk_document(
                text=parsed['text'],
                headings=parsed['headings'],
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
            if self.vector_client and small_chunk_ids:
                self._generate_embeddings(conn, chunk_result['small_chunks'], small_chunk_ids)

            return page_id

        finally:
            conn.close()

    def _generate_embeddings(self, conn, chunks: List[Dict], chunk_ids: List[int]) -> None:
        """Generate and store embeddings for chunks.

        Args:
            conn: Database connection.
            chunks: List of chunk dicts with 'content'.
            chunk_ids: List of chunk IDs in database.
        """
        try:
            texts = [c['content'] for c in chunks]

            # Batch embeddings
            embeddings = self.vector_client.get_embeddings_batch(texts)

            # Store embeddings
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                update_chunk_embedding(conn, chunk_id, embedding)

            logger.info(f"Generated {len(embeddings)} embeddings")

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
                        max_pages: int = 1000,
                        delay: float = 1.0) -> Dict[str, int]:
        """Crawl a website and index all pages.

        Args:
            start_url: Starting URL.
            allowed_domains: List of allowed domains.
            excluded_paths: List of path prefixes to exclude.
            max_pages: Maximum pages to crawl.
            delay: Delay between requests.

        Returns:
            Dict with crawl/index statistics.
        """
        crawler = Crawler(
            start_url=start_url,
            allowed_domains=allowed_domains,
            excluded_paths=excluded_paths or config.EXCLUDED_PATHS,
            max_pages=max_pages,
            delay=delay
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
