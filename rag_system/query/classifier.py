"""Query classifier module for the RAG system.

Classifies queries into types for appropriate handling.
"""

import re
from typing import Optional, Any

from rag_system.utils import get_logger

logger = get_logger(__name__)


# Query types
QUERY_TYPES = ['factual', 'procedural', 'exploratory', 'troubleshooting', 'navigational']

# Error-related keywords
ERROR_KEYWORDS = [
    'error', 'fail', 'failing', 'failed', 'broken', 'not working',
    'issue', 'problem', 'bug', 'crash', 'exception', 'timeout',
    'unable', 'cannot', 'can\'t', 'couldn\'t', 'won\'t'
]

# Classification prompt
CLASSIFICATION_PROMPT = """Classify this query into one of these categories:
- factual: Questions about what something is or specific facts
- procedural: Questions about how to do something
- exploratory: Open-ended exploration of a topic
- troubleshooting: Questions about errors or problems
- navigational: Questions about where to find something

Query: {query}

Return only the category name:"""


def detect_question_type(query: str) -> Optional[str]:
    """Detect the question word pattern.

    Args:
        query: Query text.

    Returns:
        Question type (what, how, why, where, when) or None.
    """
    query_lower = query.lower().strip()

    patterns = [
        (r'^what\b', 'what'),
        (r'^how\b', 'how'),
        (r'^why\b', 'why'),
        (r'^where\b', 'where'),
        (r'^when\b', 'when'),
    ]

    for pattern, qtype in patterns:
        if re.search(pattern, query_lower):
            return qtype

    return None


def has_error_keywords(query: str) -> bool:
    """Check if query contains error-related keywords.

    Args:
        query: Query text.

    Returns:
        True if error keywords found.
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ERROR_KEYWORDS)


class QueryClassifier:
    """Classifies queries into types."""

    def __init__(self, chat_client: Optional[Any] = None):
        """Initialize the classifier.

        Args:
            chat_client: Optional chat client for LLM classification.
        """
        self.chat_client = chat_client

    def classify(self, query: str) -> str:
        """Classify a query using rule-based approach.

        Args:
            query: Query text.

        Returns:
            Query type string.
        """
        query_lower = query.lower().strip()

        # Check for error-related queries first
        if has_error_keywords(query):
            return 'troubleshooting'

        # Check question type
        question_type = detect_question_type(query)

        if question_type == 'how':
            return 'procedural'

        if question_type == 'what':
            return 'factual'

        if question_type == 'why':
            return 'troubleshooting'

        if question_type == 'where':
            return 'navigational'

        # Check for exploratory patterns
        exploratory_patterns = [
            r'^tell me about\b',
            r'^explain\b',
            r'^describe\b',
            r'^overview\b',
        ]
        for pattern in exploratory_patterns:
            if re.search(pattern, query_lower):
                return 'exploratory'

        # Check for navigational patterns
        navigational_patterns = [
            r'\bfind\b',
            r'\blocate\b',
            r'\blocation\b',
        ]
        for pattern in navigational_patterns:
            if re.search(pattern, query_lower):
                return 'navigational'

        # Default to factual
        return 'factual'

    def classify_with_llm(self, query: str) -> str:
        """Classify query using LLM.

        Args:
            query: Query text.

        Returns:
            Query type string.
        """
        if not self.chat_client:
            return self.classify(query)

        try:
            prompt = CLASSIFICATION_PROMPT.format(query=query)
            result = self.chat_client.complete(prompt).strip().lower()

            if result in QUERY_TYPES:
                return result
            else:
                logger.warning(f"LLM returned unknown type: {result}")
                return self.classify(query)

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return self.classify(query)
