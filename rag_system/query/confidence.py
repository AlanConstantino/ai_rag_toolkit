"""Confidence scoring module for the RAG system.

Calculates confidence scores for retrieval and answer generation.
"""

from typing import Dict, List, Any

from rag_system import config
from rag_system.utils import get_logger, tokenize

logger = get_logger(__name__)


def normalize_score(score: float) -> float:
    """Normalize score to 0-1 range.

    Args:
        score: Raw score.

    Returns:
        Normalized score between 0 and 1.
    """
    return max(0.0, min(1.0, score))


def weighted_average(scores: List[float], weights: List[float]) -> float:
    """Calculate weighted average of scores.

    Args:
        scores: List of scores.
        weights: List of weights.

    Returns:
        Weighted average.
    """
    if not scores or not weights:
        return 0.0

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return weighted_sum / total_weight


def calculate_retrieval_confidence(scores: List[float]) -> float:
    """Calculate confidence based on retrieval scores.

    Args:
        scores: List of similarity/relevance scores.

    Returns:
        Retrieval confidence score.
    """
    if not scores:
        return 0.0

    # Use top score and average as indicators
    top_score = max(scores)
    avg_score = sum(scores) / len(scores)

    # Weight top score more heavily
    confidence = 0.6 * top_score + 0.4 * avg_score
    return normalize_score(confidence)


def calculate_coverage_score(query_terms: List[str], content: str) -> float:
    """Calculate how well content covers query terms.

    Args:
        query_terms: List of query terms.
        content: Content text.

    Returns:
        Coverage score between 0 and 1.
    """
    if not query_terms:
        return 0.0
    if not content:
        return 0.0

    content_lower = content.lower()
    covered = sum(1 for term in query_terms if term in content_lower)
    return covered / len(query_terms)


def get_confidence_reasons(metrics: Dict[str, float]) -> List[str]:
    """Generate human-readable reasons for confidence level.

    Args:
        metrics: Confidence metrics dict.

    Returns:
        List of reason strings.
    """
    reasons = []
    overall = metrics.get('overall', 0)
    retrieval = metrics.get('retrieval', 0)
    coverage = metrics.get('coverage', 0)

    if overall >= 0.7:
        reasons.append("High overall confidence in retrieved content")
    elif overall >= 0.4:
        reasons.append("Moderate confidence in retrieved content")
    else:
        reasons.append("Low confidence - results may not be relevant")

    if retrieval >= 0.7:
        reasons.append("Good similarity scores from retrieval")
    elif retrieval < 0.4:
        reasons.append("Low retrieval scores - content may not match well")

    if coverage >= 0.7:
        reasons.append("Query terms well covered in results")
    elif coverage < 0.4:
        reasons.append("Low coverage of query terms in results")

    return reasons


class ConfidenceAnalyzer:
    """Analyzes and scores confidence for query results."""

    RETRIEVAL_WEIGHT = 0.6
    COVERAGE_WEIGHT = 0.4

    def __init__(self, threshold: float = None):
        """Initialize the analyzer.

        Args:
            threshold: Confidence threshold for answer generation.
        """
        self.threshold = threshold if threshold is not None else config.CONFIDENCE_THRESHOLD

    def analyze(self, query: str,
                results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze confidence for query results.

        Args:
            query: Original query text.
            results: List of result dicts with 'score' and 'content'.

        Returns:
            Dict with confidence metrics.
        """
        if not results:
            return {
                'overall': 0.0,
                'retrieval': 0.0,
                'coverage': 0.0
            }

        # Calculate retrieval confidence
        scores = [r.get('score', 0) for r in results]
        retrieval_conf = calculate_retrieval_confidence(scores)

        # Calculate coverage confidence
        query_terms = tokenize(query)
        combined_content = ' '.join(r.get('content', '') for r in results)
        coverage_conf = calculate_coverage_score(query_terms, combined_content)

        # Calculate overall confidence
        overall = weighted_average(
            [retrieval_conf, coverage_conf],
            [self.RETRIEVAL_WEIGHT, self.COVERAGE_WEIGHT]
        )

        return {
            'overall': normalize_score(overall),
            'retrieval': retrieval_conf,
            'coverage': coverage_conf
        }

    def should_generate_answer(self, query: str,
                                results: List[Dict[str, Any]]) -> bool:
        """Check if confidence is high enough to generate answer.

        Args:
            query: Original query text.
            results: List of result dicts.

        Returns:
            True if answer should be generated.
        """
        metrics = self.analyze(query, results)
        return metrics['overall'] >= self.threshold

    def get_confidence_level(self, metrics: Dict[str, float]) -> str:
        """Get human-readable confidence level.

        Args:
            metrics: Confidence metrics dict.

        Returns:
            Confidence level string.
        """
        overall = metrics.get('overall', 0)
        if overall >= 0.8:
            return 'high'
        elif overall >= 0.5:
            return 'medium'
        else:
            return 'low'
