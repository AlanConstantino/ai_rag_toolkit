"""Chunker module for the RAG system.

Provides semantic chunking with heading path tracking and
parent-child chunk relationships.
"""

import re
from typing import Dict, List, Any, Optional

from rag_system import config


# =============================================================================
# Basic Text Chunking
# =============================================================================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into chunks with overlap.

    Attempts to break at sentence boundaries when possible.

    Args:
        text: Text to chunk.
        chunk_size: Target size for each chunk in characters.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end, try to find a good break point
        if end < len(text):
            # Look for sentence boundaries within the chunk
            chunk_text = text[start:end]

            # Find the last sentence boundary
            last_period = max(
                chunk_text.rfind('. '),
                chunk_text.rfind('! '),
                chunk_text.rfind('? '),
                chunk_text.rfind('.\n'),
                chunk_text.rfind('!\n'),
                chunk_text.rfind('?\n')
            )

            if last_period > chunk_size // 2:  # Only use if past halfway
                end = start + last_period + 1  # Include the punctuation
            else:
                # Try to break at word boundary
                last_space = chunk_text.rfind(' ')
                if last_space > chunk_size // 2:
                    end = start + last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position, accounting for overlap
        start = end - overlap if end < len(text) else len(text)

    return chunks


# =============================================================================
# Heading Path Building
# =============================================================================

def build_heading_path(headings: List[Dict[str, Any]], current_level: int) -> str:
    """Build a heading path string from a list of headings.

    Args:
        headings: List of heading dicts with 'level' and 'text'.
        current_level: The level to build path up to.

    Returns:
        Heading path like "Main > Section > Subsection".
    """
    if not headings:
        return ''

    # Build path from headings, keeping track of hierarchy
    path_parts = {}  # level -> heading text

    for heading in headings:
        level = heading['level']
        text = heading['text']

        # When we see a heading, clear all lower-level headings
        for l in list(path_parts.keys()):
            if l >= level:
                del path_parts[l]

        path_parts[level] = text

    # Build path up to current_level
    parts = []
    for level in sorted(path_parts.keys()):
        if level <= current_level:
            parts.append(path_parts[level])

    return ' > '.join(parts)


# =============================================================================
# Section-Based Chunking
# =============================================================================

def chunk_by_headings(text: str, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split text into sections based on headings.

    Args:
        text: Full text content.
        headings: List of heading dicts with 'level' and 'text'.

    Returns:
        List of section dicts with 'content' and 'heading_path'.
    """
    if not headings:
        return [{'content': text, 'heading_path': '', 'level': 0}]

    sections = []

    # Find heading positions in text
    heading_positions = []
    for heading in headings:
        # Search for the heading text in the document
        pattern = re.escape(heading['text'])
        match = re.search(pattern, text)
        if match:
            heading_positions.append({
                'position': match.start(),
                'heading': heading
            })

    # Sort by position
    heading_positions.sort(key=lambda x: x['position'])

    # Split into sections
    for i, hp in enumerate(heading_positions):
        start = hp['position']
        end = heading_positions[i + 1]['position'] if i + 1 < len(heading_positions) else len(text)

        content = text[start:end].strip()

        # Build heading path
        headings_so_far = [h['heading'] for h in heading_positions[:i + 1]]
        path = build_heading_path(headings_so_far, hp['heading']['level'])

        sections.append({
            'content': content,
            'heading_path': path,
            'level': hp['heading']['level']
        })

    return sections if sections else [{'content': text, 'heading_path': '', 'level': 0}]


# =============================================================================
# Semantic Chunker Class
# =============================================================================

class SemanticChunker:
    """Creates semantic chunks with heading paths and parent-child relationships."""

    def __init__(self, small_chunk_size: Optional[int] = None,
                 large_chunk_size: Optional[int] = None,
                 overlap: Optional[int] = None):
        """Initialize the chunker.

        Args:
            small_chunk_size: Size for small (retrieval) chunks.
            large_chunk_size: Size for large (context) chunks.
            overlap: Overlap between chunks.
        """
        self.small_chunk_size = small_chunk_size or config.SMALL_CHUNK_SIZE
        self.large_chunk_size = large_chunk_size or config.LARGE_CHUNK_SIZE
        self.overlap = overlap or config.CHUNK_OVERLAP

    def chunk_document(self, text: str,
                       headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk a document into large and small chunks.

        Args:
            text: Document text.
            headings: List of headings from the document.

        Returns:
            List of chunk dicts with type, content, index, heading_path, parent_index.
        """
        if not text:
            return []

        chunks = []

        # Determine heading path for the whole document
        default_path = build_heading_path(headings, max([h['level'] for h in headings], default=0)) if headings else ''

        # Create large chunks first
        large_texts = chunk_text(text, self.large_chunk_size, self.overlap)
        large_chunks = []

        for i, content in enumerate(large_texts):
            chunk = {
                'type': 'large',
                'content': content,
                'index': i,
                'heading_path': default_path,
                'parent_index': None
            }
            large_chunks.append(chunk)
            chunks.append(chunk)

        # Create small chunks and link to parent large chunks
        small_index = 0
        for large_idx, large_chunk in enumerate(large_chunks):
            small_texts = chunk_text(
                large_chunk['content'],
                self.small_chunk_size,
                self.overlap // 2
            )

            for content in small_texts:
                chunk = {
                    'type': 'small',
                    'content': content,
                    'index': small_index,
                    'heading_path': large_chunk['heading_path'],
                    'parent_index': large_idx
                }
                chunks.append(chunk)
                small_index += 1

        return chunks


# =============================================================================
# Main Function
# =============================================================================

def chunk_document(text: str, headings: List[Dict[str, Any]],
                   small_chunk_size: Optional[int] = None,
                   large_chunk_size: Optional[int] = None,
                   overlap: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Chunk a document into large and small chunks.

    Args:
        text: Document text.
        headings: List of headings from the document.
        small_chunk_size: Size for small chunks.
        large_chunk_size: Size for large chunks.
        overlap: Overlap between chunks.

    Returns:
        Dict with 'large_chunks' and 'small_chunks' lists.
    """
    if not text:
        return {'large_chunks': [], 'small_chunks': []}

    chunker = SemanticChunker(
        small_chunk_size=small_chunk_size,
        large_chunk_size=large_chunk_size,
        overlap=overlap
    )

    all_chunks = chunker.chunk_document(text, headings)

    return {
        'large_chunks': [c for c in all_chunks if c['type'] == 'large'],
        'small_chunks': [c for c in all_chunks if c['type'] == 'small']
    }
