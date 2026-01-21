"""Tests for the diversifier module."""

import unittest


class TestDiversifyResults(unittest.TestCase):
    """Test result diversification."""

    def test_diversify_limits_per_page(self):
        """diversify_results should limit results per page."""
        from rag_system.search.diversifier import diversify_results

        results = [
            (1, 0.9),  # page A
            (2, 0.8),  # page A
            (3, 0.7),  # page A
            (4, 0.6),  # page B
        ]
        chunk_to_page = {1: 'A', 2: 'A', 3: 'A', 4: 'B'}

        diversified = diversify_results(
            results, chunk_to_page,
            max_per_page=2, top_k=10
        )

        # Should only have 2 from page A
        page_a_count = sum(1 for cid, _ in diversified if chunk_to_page[cid] == 'A')
        self.assertEqual(page_a_count, 2)

    def test_diversify_respects_top_k(self):
        """diversify_results should respect top_k limit."""
        from rag_system.search.diversifier import diversify_results

        results = [(i, 1.0 - i * 0.1) for i in range(10)]
        chunk_to_page = {i: f'page{i}' for i in range(10)}

        diversified = diversify_results(
            results, chunk_to_page,
            max_per_page=5, top_k=3
        )

        self.assertEqual(len(diversified), 3)

    def test_diversify_preserves_order(self):
        """diversify_results should preserve score ordering."""
        from rag_system.search.diversifier import diversify_results

        results = [(1, 0.9), (2, 0.8), (3, 0.7)]
        chunk_to_page = {1: 'A', 2: 'B', 3: 'C'}

        diversified = diversify_results(
            results, chunk_to_page,
            max_per_page=2, top_k=10
        )

        scores = [score for _, score in diversified]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_diversify_handles_empty(self):
        """diversify_results should handle empty results."""
        from rag_system.search.diversifier import diversify_results

        result = diversify_results([], {}, max_per_page=2, top_k=10)
        self.assertEqual(result, [])


class TestDiversifier(unittest.TestCase):
    """Test Diversifier class."""

    def test_diversifier_with_database(self):
        """Diversifier should load chunk-page mapping from database."""
        from rag_system.search.diversifier import Diversifier
        import tempfile
        import os
        from rag_system.database import init_db, get_connection, insert_page, insert_chunk

        temp_fd, temp_path = tempfile.mkstemp(suffix='.db')
        os.close(temp_fd)
        init_db(temp_path)

        conn = get_connection(temp_path)
        page1_id = insert_page(conn, url='http://test.com/1', title='Page 1',
                              raw_html='', parsed_text='', content_hash='a')
        page2_id = insert_page(conn, url='http://test.com/2', title='Page 2',
                              raw_html='', parsed_text='', content_hash='b')
        chunk1_id = insert_chunk(conn, page_id=page1_id, chunk_type='small',
                                chunk_index=0, content='test', heading_path='')
        chunk2_id = insert_chunk(conn, page_id=page1_id, chunk_type='small',
                                chunk_index=1, content='test', heading_path='')
        chunk3_id = insert_chunk(conn, page_id=page2_id, chunk_type='small',
                                chunk_index=0, content='test', heading_path='')
        conn.close()

        try:
            diversifier = Diversifier(temp_path)
            results = [(chunk1_id, 0.9), (chunk2_id, 0.8), (chunk3_id, 0.7)]

            diversified = diversifier.diversify(results, max_per_page=1, top_k=10)

            # Should only have 1 from each page
            self.assertEqual(len(diversified), 2)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
