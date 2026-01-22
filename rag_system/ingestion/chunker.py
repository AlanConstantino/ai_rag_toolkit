"""Chunker module for the RAG system.

Provides semantic chunking with heading path tracking and
parent-child chunk relationships.
"""

import re
from typing import Dict, List, Any, Optional

from rag_system import config


# =============================================================================
# Token Estimation
# =============================================================================

def estimate_token_count(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses a simple heuristic based on word boundaries and punctuation.
    This provides a reasonable approximation without requiring external
    tokenizer libraries like tiktoken.

    Heuristic: Split on whitespace and common punctuation patterns to
    approximate how LLM tokenizers split text. Most tokenizers split on
    whitespace, punctuation, and within long words.

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if not text or not text.strip():
        return 0

    words = text.split()
    punctuation = '.,!?;:\'"()[]{}/<>-_@#$%^&*+=~`|'
    token_count = 0

    for word in words:
        if not word:
            continue

        # Each word is at least one token
        token_count += 1

        # Count punctuation at both ends of the word
        # Common patterns: "hello," -> 2 tokens, "world!" -> 2 tokens
        punct_count = 0
        for char in word:
            if char in punctuation:
                punct_count += 1
            else:
                break

        for char in reversed(word):
            if char in punctuation:
                punct_count += 1
            else:
                break

        # Add punctuation tokens (but avoid double-counting all-punctuation words)
        if punct_count < len(word):
            token_count += punct_count

        # Long words often get split by tokenizers
        # Add approximately 1 token per 5 characters beyond 10
        word_no_punct = word.strip(punctuation)
        if len(word_no_punct) > 10:
            token_count += (len(word_no_punct) - 10) // 5

    return token_count


# =============================================================================
# Basic Text Chunking
# =============================================================================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100,
               use_tokens: bool = False) -> List[str]:
    """Split text into chunks with overlap.

    Attempts to break at sentence boundaries when possible.

    Args:
        text: Text to chunk.
        chunk_size: Target size for each chunk (characters or tokens).
        overlap: Amount to overlap between chunks (characters or tokens).
        use_tokens: If True, chunk_size and overlap are in tokens, not characters.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    # For token mode, check size differently
    if use_tokens:
        if estimate_token_count(text) <= chunk_size:
            return [text]
    else:
        if len(text) <= chunk_size:
            return [text]

    if use_tokens:
        return _chunk_by_tokens(text, chunk_size, overlap)
    else:
        return _chunk_by_characters(text, chunk_size, overlap)


def _chunk_by_characters(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into chunks by character count."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end, try to find a good break point
        if end < len(text):
            candidate_chunk = text[start:end]

            # Find the last sentence boundary
            sentence_markers = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
            last_sentence = max(candidate_chunk.rfind(marker) for marker in sentence_markers)

            if last_sentence > chunk_size // 2:
                # Break at sentence boundary
                end = start + last_sentence + 1
            else:
                # Break at word boundary
                last_space = candidate_chunk.rfind(' ')
                if last_space > chunk_size // 2:
                    end = start + last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position, accounting for overlap
        start = end - overlap if end < len(text) else len(text)

    return chunks


def _chunk_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """Split text into chunks by token count.

    Uses word-level splitting with token estimation to create chunks
    that fit within the token limit while respecting sentence boundaries.
    """
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    chunks = []
    current_chunk_words = []
    current_token_count = 0

    for word in words:
        word_tokens = estimate_token_count(word)

        # Check if adding this word would exceed the limit
        if current_token_count + word_tokens > max_tokens and current_chunk_words:
            # Save current chunk
            chunk_text = ' '.join(current_chunk_words)
            chunks.append(chunk_text)

            # Build overlap from end of current chunk
            overlap_words = []
            overlap_count = 0
            for w in reversed(current_chunk_words):
                w_tokens = estimate_token_count(w)
                if overlap_count + w_tokens <= overlap_tokens:
                    overlap_words.append(w)
                    overlap_count += w_tokens
                else:
                    break

            # Reverse to maintain original order
            current_chunk_words = list(reversed(overlap_words))
            current_token_count = overlap_count

        current_chunk_words.append(word)
        current_token_count += word_tokens

    # Add the last chunk
    if current_chunk_words:
        chunks.append(' '.join(current_chunk_words))

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
        """Chunk a document into large and small chunks with proper heading paths.

        First splits text by headings to preserve section structure, then creates
        large and small chunks within each section with the correct heading path.

        Args:
            text: Document text.
            headings: List of headings from the document.

        Returns:
            List of chunk dicts with type, content, index, heading_path, parent_index.
        """
        if not text:
            return []

        chunks = []
        large_chunks = []
        large_index = 0
        small_index = 0

        # Split document by headings first to get proper section boundaries
        sections = chunk_by_headings(text, headings)

        for section in sections:
            section_content = section['content']
            section_path = section['heading_path']

            # Skip empty sections
            if not section_content.strip():
                continue

            # Create large chunks for this section
            large_texts = chunk_text(section_content, self.large_chunk_size, self.overlap)

            for content in large_texts:
                if not content.strip():
                    continue

                large_chunk = {
                    'type': 'large',
                    'content': content,
                    'index': large_index,
                    'heading_path': section_path,
                    'parent_index': None
                }
                large_chunks.append(large_chunk)
                chunks.append(large_chunk)
                large_index += 1

        # Create small chunks and link to parent large chunks
        for large_idx, large_chunk in enumerate(large_chunks):
            small_texts = chunk_text(
                large_chunk['content'],
                self.small_chunk_size,
                self.overlap // 2
            )

            for content in small_texts:
                if not content.strip():
                    continue

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
# Markdown-Based Chunking
# =============================================================================

def chunk_markdown(markdown: str,
                   small_chunk_size: Optional[int] = None,
                   large_chunk_size: Optional[int] = None,
                   overlap: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Chunk Markdown text based on its heading structure.

    This is the preferred chunking method as Markdown headings are unambiguous
    content structure (unlike HTML where navigation headings can pollute results).

    Args:
        markdown: Markdown text to chunk.
        small_chunk_size: Size for small chunks.
        large_chunk_size: Size for large chunks.
        overlap: Overlap between chunks.

    Returns:
        Dict with 'large_chunks' and 'small_chunks' lists.
    """
    from rag_system.ingestion.html_to_markdown import split_markdown_by_headings

    if not markdown:
        return {'large_chunks': [], 'small_chunks': []}

    small_size = small_chunk_size or config.SMALL_CHUNK_SIZE
    large_size = large_chunk_size or config.LARGE_CHUNK_SIZE
    chunk_overlap = overlap or config.CHUNK_OVERLAP

    # Split markdown by headings
    sections = split_markdown_by_headings(markdown)

    chunks = []
    large_chunks = []
    large_index = 0
    small_index = 0

    for section in sections:
        section_content = section['content']
        section_path = section['heading_path']

        if not section_content.strip():
            continue

        # Create large chunks for this section
        large_texts = chunk_text(section_content, large_size, chunk_overlap)

        for content in large_texts:
            if not content.strip():
                continue

            large_chunk = {
                'type': 'large',
                'content': content,
                'index': large_index,
                'heading_path': section_path,
                'parent_index': None
            }
            large_chunks.append(large_chunk)
            chunks.append(large_chunk)
            large_index += 1

    # Create small chunks linked to parent large chunks
    for large_idx, large_chunk in enumerate(large_chunks):
        small_texts = chunk_text(
            large_chunk['content'],
            small_size,
            chunk_overlap // 2
        )

        for content in small_texts:
            if not content.strip():
                continue

            chunk = {
                'type': 'small',
                'content': content,
                'index': small_index,
                'heading_path': large_chunk['heading_path'],
                'parent_index': large_idx
            }
            chunks.append(chunk)
            small_index += 1

    return {
        'large_chunks': [c for c in chunks if c['type'] == 'large'],
        'small_chunks': [c for c in chunks if c['type'] == 'small']
    }


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
