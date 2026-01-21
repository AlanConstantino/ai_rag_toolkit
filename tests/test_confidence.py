"""Tests for the confidence scoring module."""

import unittest


class TestConfidenceScoring(unittest.TestCase):
    """Test confidence score calculation."""

    def test_calculate_retrieval_confidence(self):
        """calculate_retrieval_confidence should score based on scores."""
        from rag_system.query.confidence import calculate_retrieval_confidence

        # High scores = high confidence
        scores = [0.9, 0.85, 0.8]
        confidence = calculate_retrieval_confidence(scores)
        self.assertGreater(confidence, 0.7)

        # Low scores = low confidence
        scores = [0.3, 0.2, 0.1]
        confidence = calculate_retrieval_confidence(scores)
        self.assertLess(confidence, 0.5)

    def test_calculate_retrieval_confidence_empty(self):
        """calculate_retrieval_confidence should handle empty scores."""
        from rag_system.query.confidence import calculate_retrieval_confidence

        confidence = calculate_retrieval_confidence([])
        self.assertEqual(confidence, 0.0)

    def test_calculate_coverage_score(self):
        """calculate_coverage_score should measure query term coverage."""
        from rag_system.query.confidence import calculate_coverage_score

        # All terms covered
        query_terms = ['redis', 'cache']
        content = "Redis is a fast in-memory cache system"
        score = calculate_coverage_score(query_terms, content)
        self.assertEqual(score, 1.0)

        # Partial coverage
        query_terms = ['redis', 'kafka']
        content = "Redis is a cache"
        score = calculate_coverage_score(query_terms, content)
        self.assertEqual(score, 0.5)

    def test_calculate_coverage_score_empty(self):
        """calculate_coverage_score should handle empty inputs."""
        from rag_system.query.confidence import calculate_coverage_score

        self.assertEqual(calculate_coverage_score([], "content"), 0.0)
        self.assertEqual(calculate_coverage_score(['term'], ""), 0.0)


class TestConfidenceAnalyzer(unittest.TestCase):
    """Test ConfidenceAnalyzer class."""

    def test_analyze_returns_confidence(self):
        """analyze should return confidence metrics."""
        from rag_system.query.confidence import ConfidenceAnalyzer

        analyzer = ConfidenceAnalyzer()
        results = [
            {'score': 0.9, 'content': 'Redis is a cache'},
            {'score': 0.8, 'content': 'Redis configuration'},
        ]
        metrics = analyzer.analyze("redis cache", results)

        self.assertIn('overall', metrics)
        self.assertIn('retrieval', metrics)
        self.assertIn('coverage', metrics)
        self.assertGreater(metrics['overall'], 0)

    def test_analyze_empty_results(self):
        """analyze should handle empty results."""
        from rag_system.query.confidence import ConfidenceAnalyzer

        analyzer = ConfidenceAnalyzer()
        metrics = analyzer.analyze("redis cache", [])

        self.assertEqual(metrics['overall'], 0.0)
        self.assertEqual(metrics['retrieval'], 0.0)

    def test_should_generate_answer(self):
        """should_generate_answer should check threshold."""
        from rag_system.query.confidence import ConfidenceAnalyzer

        analyzer = ConfidenceAnalyzer(threshold=0.5)

        # High confidence = should generate
        results = [{'score': 0.9, 'content': 'Redis cache'}]
        self.assertTrue(analyzer.should_generate_answer("redis", results))

        # Low confidence = should not generate
        results = [{'score': 0.2, 'content': 'unrelated'}]
        self.assertFalse(analyzer.should_generate_answer("redis", results))


class TestScoreNormalization(unittest.TestCase):
    """Test score normalization."""

    def test_normalize_score(self):
        """normalize_score should clamp to 0-1 range."""
        from rag_system.query.confidence import normalize_score

        self.assertEqual(normalize_score(0.5), 0.5)
        self.assertEqual(normalize_score(-0.5), 0.0)
        self.assertEqual(normalize_score(1.5), 1.0)

    def test_weighted_average(self):
        """weighted_average should combine scores correctly."""
        from rag_system.query.confidence import weighted_average

        scores = [0.8, 0.6]
        weights = [0.7, 0.3]
        avg = weighted_average(scores, weights)

        expected = (0.8 * 0.7 + 0.6 * 0.3)
        self.assertAlmostEqual(avg, expected)

    def test_weighted_average_empty(self):
        """weighted_average should handle empty inputs."""
        from rag_system.query.confidence import weighted_average

        self.assertEqual(weighted_average([], []), 0.0)


class TestConfidenceReasons(unittest.TestCase):
    """Test confidence reason generation."""

    def test_get_confidence_reasons_high(self):
        """get_confidence_reasons should explain high confidence."""
        from rag_system.query.confidence import get_confidence_reasons

        metrics = {'overall': 0.9, 'retrieval': 0.9, 'coverage': 0.9}
        reasons = get_confidence_reasons(metrics)

        self.assertTrue(len(reasons) > 0)
        self.assertTrue(any('high' in r.lower() or 'good' in r.lower() for r in reasons))

    def test_get_confidence_reasons_low(self):
        """get_confidence_reasons should explain low confidence."""
        from rag_system.query.confidence import get_confidence_reasons

        metrics = {'overall': 0.2, 'retrieval': 0.2, 'coverage': 0.2}
        reasons = get_confidence_reasons(metrics)

        self.assertTrue(len(reasons) > 0)
        self.assertTrue(any('low' in r.lower() for r in reasons))


if __name__ == '__main__':
    unittest.main()
