"""Summarizer module for the RAG system.

Provides hierarchical summarization at page, system, and global levels.
"""

from typing import Dict, List, Any, Optional

from rag_system.database import get_connection, update_page_summary, get_all_pages
from rag_system.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Prompt Builders
# =============================================================================

def build_page_summary_prompt(text: str) -> str:
    """Build prompt for page summarization.

    Args:
        text: Page text content.

    Returns:
        Prompt string.
    """
    return f"""Summarize this documentation page in 2-3 sentences.
Focus on the main topic, purpose, and key information.

TEXT:
{text}

Provide a concise summary:"""


def build_system_summary_prompt(system_name: str,
                                 pages: List[Dict[str, str]]) -> str:
    """Build prompt for system summarization.

    Args:
        system_name: Name of the system.
        pages: List of page dicts with 'title' and 'summary'.

    Returns:
        Prompt string.
    """
    page_list = "\n".join(
        f"- {p['title']}: {p['summary']}"
        for p in pages
    )

    return f"""Create a comprehensive overview of {system_name} based on these page summaries.
Focus on what {system_name} is, its purpose, and key capabilities.

PAGES:
{page_list}

Provide a 3-4 sentence overview:"""


def build_global_summary_prompt(systems: List[Dict[str, str]]) -> str:
    """Build prompt for global summary.

    Args:
        systems: List of system dicts with 'name' and 'summary'.

    Returns:
        Prompt string.
    """
    system_list = "\n".join(
        f"- {s['name']}: {s['summary']}"
        for s in systems
    )

    return f"""Create a high-level overview of this documentation covering these systems.
Explain what the documentation covers and how the systems relate.

SYSTEMS:
{system_list}

Provide a 2-3 sentence overview:"""


# =============================================================================
# Summarizers
# =============================================================================

class PageSummarizer:
    """Summarizes individual pages."""

    def __init__(self, chat_client: Optional[Any] = None,
                 max_text_length: int = 5000):
        """Initialize the page summarizer.

        Args:
            chat_client: Chat API client for LLM calls.
            max_text_length: Maximum text length to send to LLM.
        """
        self.chat_client = chat_client
        self.max_text_length = max_text_length

    def summarize(self, text: str) -> str:
        """Summarize page text.

        Args:
            text: Page text content.

        Returns:
            Summary string.
        """
        if not text or not text.strip():
            return ""

        if not self.chat_client:
            logger.warning("No chat client configured for summarization")
            return ""

        try:
            # Truncate long text
            truncated = text[:self.max_text_length]
            prompt = build_page_summary_prompt(truncated)
            return self.chat_client.complete(prompt)
        except Exception as e:
            logger.error(f"Page summarization failed: {e}")
            return ""


class SystemSummarizer:
    """Summarizes systems based on page summaries."""

    def __init__(self, chat_client: Optional[Any] = None):
        """Initialize the system summarizer.

        Args:
            chat_client: Chat API client for LLM calls.
        """
        self.chat_client = chat_client

    def summarize(self, system_name: str,
                  page_summaries: List[Dict[str, str]]) -> str:
        """Summarize a system.

        Args:
            system_name: Name of the system.
            page_summaries: List of page dicts with 'title' and 'summary'.

        Returns:
            Summary string.
        """
        if not page_summaries:
            return ""

        if not self.chat_client:
            logger.warning("No chat client configured for summarization")
            return ""

        try:
            prompt = build_system_summary_prompt(system_name, page_summaries)
            return self.chat_client.complete(prompt)
        except Exception as e:
            logger.error(f"System summarization failed: {e}")
            return ""


class GlobalSummarizer:
    """Generates global documentation summary."""

    def __init__(self, chat_client: Optional[Any] = None):
        """Initialize the global summarizer.

        Args:
            chat_client: Chat API client for LLM calls.
        """
        self.chat_client = chat_client

    def summarize(self, system_summaries: List[Dict[str, str]]) -> str:
        """Generate global summary.

        Args:
            system_summaries: List of system dicts with 'name' and 'summary'.

        Returns:
            Summary string.
        """
        if not system_summaries:
            return ""

        if not self.chat_client:
            logger.warning("No chat client configured for summarization")
            return ""

        try:
            prompt = build_global_summary_prompt(system_summaries)
            return self.chat_client.complete(prompt)
        except Exception as e:
            logger.error(f"Global summarization failed: {e}")
            return ""


# =============================================================================
# Pipeline
# =============================================================================

class SummarizationPipeline:
    """Orchestrates hierarchical summarization."""

    def __init__(self, db_path: str, chat_client: Optional[Any] = None):
        """Initialize the pipeline.

        Args:
            db_path: Path to the SQLite database.
            chat_client: Chat API client for LLM calls.
        """
        self.db_path = db_path
        self.chat_client = chat_client
        self.page_summarizer = PageSummarizer(chat_client)
        self.system_summarizer = SystemSummarizer(chat_client)
        self.global_summarizer = GlobalSummarizer(chat_client)

    def summarize_all_pages(self) -> List[Dict[str, Any]]:
        """Summarize all pages in database.

        Returns:
            List of page dicts with summaries.
        """
        conn = get_connection(self.db_path)
        try:
            pages = get_all_pages(conn)
            results = []

            for page in pages:
                text = page.get('parsed_text', '')
                summary = self.page_summarizer.summarize(text)

                if summary:
                    update_page_summary(conn, page['id'], summary)

                results.append({
                    'id': page['id'],
                    'title': page.get('title', ''),
                    'summary': summary
                })

            return results
        finally:
            conn.close()

    def get_page_summaries(self) -> List[Dict[str, Any]]:
        """Get all page summaries from database.

        Returns:
            List of page dicts with summaries.
        """
        conn = get_connection(self.db_path)
        try:
            pages = get_all_pages(conn)
            return [
                {
                    'id': page['id'],
                    'title': page.get('title', ''),
                    'url': page.get('url', ''),
                    'summary': page.get('summary', '')
                }
                for page in pages
            ]
        finally:
            conn.close()

    def summarize_system(self, system_name: str,
                         page_ids: List[int]) -> str:
        """Summarize a system based on its pages.

        Args:
            system_name: Name of the system.
            page_ids: List of page IDs belonging to system.

        Returns:
            System summary string.
        """
        conn = get_connection(self.db_path)
        try:
            page_summaries = []
            for page_id in page_ids:
                cursor = conn.execute(
                    "SELECT title, summary FROM pages WHERE id = ?",
                    (page_id,)
                )
                row = cursor.fetchone()
                if row and row['summary']:
                    page_summaries.append({
                        'title': row['title'] or '',
                        'summary': row['summary']
                    })

            return self.system_summarizer.summarize(system_name, page_summaries)
        finally:
            conn.close()

    def generate_global_summary(self,
                                 systems: List[Dict[str, str]]) -> str:
        """Generate global documentation summary.

        Args:
            systems: List of system dicts with 'name' and 'summary'.

        Returns:
            Global summary string.
        """
        return self.global_summarizer.summarize(systems)

    def detect_systems_from_urls(self) -> Dict[str, List[int]]:
        """Detect systems based on URL patterns.

        Groups pages by their first path segment to identify distinct
        documentation systems/sections.

        Returns:
            Dict mapping system names to lists of page IDs.
        """
        from urllib.parse import urlparse

        conn = get_connection(self.db_path)
        try:
            pages = get_all_pages(conn)
            systems: Dict[str, List[int]] = {}

            for page in pages:
                url = page.get('url', '')
                if not url:
                    continue

                parsed = urlparse(url)
                path_parts = [p for p in parsed.path.split('/') if p]

                # Use first meaningful path segment as system name
                if path_parts:
                    # Skip common prefixes like 'docs', 'api', 'en'
                    system_name = path_parts[0]
                    if system_name in ('docs', 'api', 'en', 'latest', 'stable'):
                        system_name = path_parts[1] if len(path_parts) > 1 else parsed.netloc
                else:
                    # No path - use domain as system name
                    system_name = parsed.netloc

                # Clean up the system name
                system_name = system_name.replace('-', ' ').replace('_', ' ').title()

                if system_name not in systems:
                    systems[system_name] = []
                systems[system_name].append(page['id'])

            return systems
        finally:
            conn.close()

    def rebuild_all_summaries(self) -> Dict[str, Any]:
        """Rebuild all summaries: page, system, and global.

        This is the main entry point for the rebuild-summaries command.

        Returns:
            Dict with statistics about the rebuild operation.
        """
        from rag_system.database import (
            insert_system, update_system_summary, get_system_by_name,
            set_global_summary
        )

        stats = {
            'pages_summarized': 0,
            'systems_created': 0,
            'systems_summarized': 0,
            'global_summary_generated': False,
            'errors': []
        }

        conn = get_connection(self.db_path)
        try:
            # Step 1: Summarize all pages that don't have summaries
            logger.info("Step 1: Summarizing pages...")
            pages = get_all_pages(conn)
            for page in pages:
                if not page.get('summary'):
                    text = page.get('parsed_text', '')
                    if text:
                        try:
                            summary = self.page_summarizer.summarize(text)
                            if summary:
                                update_page_summary(conn, page['id'], summary)
                                stats['pages_summarized'] += 1
                        except Exception as e:
                            stats['errors'].append(f"Page {page['id']}: {e}")

            # Step 2: Detect and create systems
            logger.info("Step 2: Detecting systems from URLs...")
            detected_systems = self.detect_systems_from_urls()
            logger.info(f"Detected {len(detected_systems)} systems")

            # Step 3: Create/update systems and generate summaries
            logger.info("Step 3: Generating system summaries...")
            system_summaries = []

            for system_name, page_ids in detected_systems.items():
                try:
                    # Check if system exists
                    existing = get_system_by_name(conn, system_name)
                    if existing:
                        system_id = existing['id']
                    else:
                        # Create new system with empty description and summary
                        system_id = insert_system(
                            conn, system_name,
                            description='',
                            summary=''
                        )
                        stats['systems_created'] += 1

                    # Link pages to system
                    for page_id in page_ids:
                        conn.execute(
                            "INSERT OR IGNORE INTO page_systems (page_id, system_id) VALUES (?, ?)",
                            (page_id, system_id)
                        )
                    conn.commit()

                    # Generate system summary
                    summary = self.summarize_system(system_name, page_ids)
                    if summary:
                        update_system_summary(conn, system_id, summary)
                        stats['systems_summarized'] += 1
                        system_summaries.append({
                            'name': system_name,
                            'summary': summary
                        })
                except Exception as e:
                    stats['errors'].append(f"System {system_name}: {e}")

            # Step 4: Generate global summary
            logger.info("Step 4: Generating global summary...")
            if system_summaries:
                try:
                    global_summary = self.generate_global_summary(system_summaries)
                    if global_summary:
                        set_global_summary(conn, global_summary)
                        stats['global_summary_generated'] = True
                except Exception as e:
                    stats['errors'].append(f"Global summary: {e}")

            return stats
        finally:
            conn.close()
