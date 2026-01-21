"""Tests for the chunker module."""

import unittest


class TestTextChunking(unittest.TestCase):
    """Test basic text chunking functionality."""

    def test_chunk_text_basic(self):
        """chunk_text should split text into chunks."""
        from rag_system.ingestion.chunker import chunk_text

        text = "This is a test. " * 100  # Create text longer than chunk size

        chunks = chunk_text(text, chunk_size=200, overlap=50)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 250)  # Allow some overflow for words

    def test_chunk_text_short_text(self):
        """chunk_text should return single chunk for short text."""
        from rag_system.ingestion.chunker import chunk_text

        text = "Short text."

        chunks = chunk_text(text, chunk_size=500, overlap=100)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Short text.")

    def test_chunk_text_overlap(self):
        """chunk_text should create overlapping chunks."""
        from rag_system.ingestion.chunker import chunk_text

        # Create predictable text
        text = " ".join([f"word{i}" for i in range(50)])

        chunks = chunk_text(text, chunk_size=100, overlap=20)

        # Check that consecutive chunks share some content
        if len(chunks) > 1:
            chunk1_words = set(chunks[0].split())
            chunk2_words = set(chunks[1].split())
            # There should be some overlap
            overlap = chunk1_words & chunk2_words
            self.assertGreater(len(overlap), 0)

    def test_chunk_text_respects_sentences(self):
        """chunk_text should try to break at sentence boundaries."""
        from rag_system.ingestion.chunker import chunk_text

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = chunk_text(text, chunk_size=35, overlap=0)

        # Chunks should end with complete sentences when possible
        for chunk in chunks[:-1]:  # Last chunk might not end with period
            self.assertTrue(
                chunk.endswith('.') or chunk.endswith('!') or chunk.endswith('?'),
                f"Chunk should end with sentence boundary: {chunk}"
            )


class TestHeadingPaths(unittest.TestCase):
    """Test heading path extraction functionality."""

    def test_build_heading_path(self):
        """build_heading_path should create path from headings."""
        from rag_system.ingestion.chunker import build_heading_path

        headings = [
            {'level': 1, 'text': 'Main'},
            {'level': 2, 'text': 'Section'},
            {'level': 3, 'text': 'Subsection'}
        ]

        path = build_heading_path(headings, current_level=3)

        self.assertEqual(path, 'Main > Section > Subsection')

    def test_build_heading_path_resets_on_higher_level(self):
        """build_heading_path should reset lower levels on higher heading."""
        from rag_system.ingestion.chunker import build_heading_path

        headings = [
            {'level': 1, 'text': 'First'},
            {'level': 2, 'text': 'Sub'},
            {'level': 1, 'text': 'Second'},  # Reset
            {'level': 2, 'text': 'NewSub'}
        ]

        path = build_heading_path(headings, current_level=2)

        self.assertEqual(path, 'Second > NewSub')

    def test_build_heading_path_empty(self):
        """build_heading_path should return empty string for no headings."""
        from rag_system.ingestion.chunker import build_heading_path

        path = build_heading_path([], current_level=1)

        self.assertEqual(path, '')


class TestSemanticChunker(unittest.TestCase):
    """Test SemanticChunker class."""

    def test_chunker_creates_large_and_small_chunks(self):
        """SemanticChunker should create both large and small chunks."""
        from rag_system.ingestion.chunker import SemanticChunker

        chunker = SemanticChunker(
            small_chunk_size=100,
            large_chunk_size=300,
            overlap=20
        )

        text = "Introduction paragraph. " * 50
        headings = [{'level': 1, 'text': 'Title'}]

        chunks = chunker.chunk_document(text, headings)

        large_chunks = [c for c in chunks if c['type'] == 'large']
        small_chunks = [c for c in chunks if c['type'] == 'small']

        self.assertGreater(len(large_chunks), 0)
        self.assertGreater(len(small_chunks), 0)

    def test_chunker_assigns_heading_paths(self):
        """SemanticChunker should assign heading paths to chunks."""
        from rag_system.ingestion.chunker import SemanticChunker

        chunker = SemanticChunker(
            small_chunk_size=50,
            large_chunk_size=100,
            overlap=10
        )

        text = "Content under heading."
        headings = [
            {'level': 1, 'text': 'Main'},
            {'level': 2, 'text': 'Section'}
        ]

        chunks = chunker.chunk_document(text, headings)

        for chunk in chunks:
            self.assertIn('heading_path', chunk)
            if chunk['heading_path']:
                self.assertIn('Main', chunk['heading_path'])

    def test_chunker_links_small_to_large(self):
        """SemanticChunker should link small chunks to parent large chunks."""
        from rag_system.ingestion.chunker import SemanticChunker

        chunker = SemanticChunker(
            small_chunk_size=50,
            large_chunk_size=200,
            overlap=10
        )

        text = "Word " * 100  # Long enough for multiple chunks

        chunks = chunker.chunk_document(text, [])

        small_chunks = [c for c in chunks if c['type'] == 'small']
        large_chunks = [c for c in chunks if c['type'] == 'large']

        # Each small chunk should have a parent_index pointing to a large chunk
        for small in small_chunks:
            self.assertIn('parent_index', small)
            # Parent index should be valid
            if small['parent_index'] is not None:
                self.assertLess(small['parent_index'], len(large_chunks))

    def test_chunker_includes_content_and_index(self):
        """SemanticChunker chunks should have content and index."""
        from rag_system.ingestion.chunker import SemanticChunker

        chunker = SemanticChunker()

        text = "Test content."
        chunks = chunker.chunk_document(text, [])

        for chunk in chunks:
            self.assertIn('content', chunk)
            self.assertIn('index', chunk)
            self.assertIn('type', chunk)
            self.assertIsInstance(chunk['content'], str)
            self.assertIsInstance(chunk['index'], int)


class TestSectionChunking(unittest.TestCase):
    """Test section-based chunking."""

    def test_chunk_by_headings(self):
        """chunk_by_headings should split text at heading boundaries."""
        from rag_system.ingestion.chunker import chunk_by_headings

        text = """# Introduction
This is the intro.

## Section 1
Content of section 1.

## Section 2
Content of section 2.
"""
        headings = [
            {'level': 1, 'text': 'Introduction'},
            {'level': 2, 'text': 'Section 1'},
            {'level': 2, 'text': 'Section 2'}
        ]

        sections = chunk_by_headings(text, headings)

        self.assertEqual(len(sections), 3)

    def test_chunk_by_headings_no_headings(self):
        """chunk_by_headings should return single section for no headings."""
        from rag_system.ingestion.chunker import chunk_by_headings

        text = "Just plain text without any headings."

        sections = chunk_by_headings(text, [])

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]['content'], text)


class TestChunkDocument(unittest.TestCase):
    """Test the main chunk_document function."""

    def test_chunk_document_returns_structured_data(self):
        """chunk_document should return properly structured chunk data."""
        from rag_system.ingestion.chunker import chunk_document

        text = "Sample text content. " * 20
        headings = [{'level': 1, 'text': 'Title'}]

        result = chunk_document(
            text=text,
            headings=headings,
            small_chunk_size=100,
            large_chunk_size=300
        )

        self.assertIn('large_chunks', result)
        self.assertIn('small_chunks', result)

        # Verify structure
        for chunk in result['large_chunks']:
            self.assertIn('content', chunk)
            self.assertIn('index', chunk)
            self.assertIn('heading_path', chunk)

        for chunk in result['small_chunks']:
            self.assertIn('content', chunk)
            self.assertIn('index', chunk)
            self.assertIn('parent_index', chunk)

    def test_chunk_document_empty_text(self):
        """chunk_document should handle empty text."""
        from rag_system.ingestion.chunker import chunk_document

        result = chunk_document(text='', headings=[])

        self.assertEqual(result['large_chunks'], [])
        self.assertEqual(result['small_chunks'], [])


if __name__ == '__main__':
    unittest.main()
