"""Answer generator module for the RAG system.

Generates answers from context using LLM.
"""

import re
from typing import Dict, List, Any, Optional

from rag_system.utils import get_logger

logger = get_logger(__name__)


# Answer prompt templates by query type
PROMPT_TEMPLATES = {
    'factual': """Based on the following context, provide a factual answer to the question.
Be concise and accurate.

CONTEXT:
{context}

QUESTION: {query}

Answer:""",

    'procedural': """Based on the following context, provide step-by-step instructions.
Be clear and actionable.

CONTEXT:
{context}

QUESTION: {query}

Steps:""",

    'exploratory': """Based on the following context, provide a comprehensive explanation.
Cover the key concepts and relationships.

CONTEXT:
{context}

QUESTION: {query}

Explanation:""",

    'troubleshooting': """Based on the following context, help diagnose and solve the issue.
Identify potential causes and solutions.

CONTEXT:
{context}

QUESTION: {query}

Analysis:""",

    'navigational': """Based on the following context, help locate the requested information.
Point to specific sections or resources.

CONTEXT:
{context}

QUESTION: {query}

Location:""",

    'default': """Based on the following context, answer the question.

CONTEXT:
{context}

QUESTION: {query}

Answer:"""
}

# Prefixes to remove from answers
ANSWER_PREFIXES = [
    'Answer:', 'Response:', 'Here is', 'Sure,', 'Based on the context,',
    'According to the documentation,', 'Steps:', 'Explanation:', 'Analysis:',
    'Location:'
]


def build_answer_prompt(query: str, context: str,
                        query_type: str = 'default') -> str:
    """Build prompt for answer generation.

    Args:
        query: Original query.
        context: Retrieved context.
        query_type: Type of query for template selection.

    Returns:
        Complete prompt string.
    """
    template = PROMPT_TEMPLATES.get(query_type, PROMPT_TEMPLATES['default'])
    return template.format(query=query, context=context)


def format_answer(answer: str,
                  sources: Optional[List[Dict[str, str]]] = None) -> str:
    """Format answer for display.

    Args:
        answer: Raw answer text.
        sources: Optional list of source dicts with 'title' and 'url'.

    Returns:
        Formatted answer string.
    """
    # Clean whitespace
    formatted = answer.strip()

    # Remove common prefixes
    for prefix in ANSWER_PREFIXES:
        if formatted.lower().startswith(prefix.lower()):
            formatted = formatted[len(prefix):].strip()

    # Add sources if provided
    if sources:
        formatted += "\n\nSources:"
        for source in sources:
            title = source.get('title', 'Unknown')
            url = source.get('url', '')
            if url:
                formatted += f"\n- {title} ({url})"
            else:
                formatted += f"\n- {title}"

    return formatted


def get_fallback_response(query: str) -> str:
    """Generate fallback response when answer not possible.

    Args:
        query: Original query.

    Returns:
        Fallback response string.
    """
    # Extract key terms from query
    terms = re.findall(r'\b\w{3,}\b', query)
    key_terms = [t for t in terms if t.lower() not in {
        'what', 'how', 'where', 'when', 'why', 'the', 'and', 'for'
    }]

    if key_terms:
        topic = ' '.join(key_terms[:2])
        return (f"I don't have enough information to answer your question about {topic}. "
                f"Try rephrasing your question or searching for more specific terms.")
    else:
        return ("I don't have enough information to answer that question. "
                "Try rephrasing or asking about a specific topic.")


class AnswerGenerator:
    """Generates answers from context using LLM."""

    CONFIDENCE_DISCLAIMER_THRESHOLD = 0.5

    def __init__(self, chat_client: Optional[Any] = None):
        """Initialize the generator.

        Args:
            chat_client: Chat API client for LLM calls.
        """
        self.chat_client = chat_client

    def generate(self, query: str, context: str,
                 query_type: str = 'default') -> str:
        """Generate answer from context.

        Args:
            query: Original query.
            context: Retrieved context.
            query_type: Type of query.

        Returns:
            Generated answer string.
        """
        if not context.strip():
            return "I don't have enough information to answer that question."

        if not self.chat_client:
            return "Answer generation is not configured."

        try:
            prompt = build_answer_prompt(query, context, query_type)
            raw_answer = self.chat_client.complete(prompt)
            return format_answer(raw_answer)

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "I was unable to generate an answer. Please try again."

    def generate_with_confidence(self, query: str, context: str,
                                  confidence: float,
                                  query_type: str = 'default') -> Dict[str, Any]:
        """Generate answer with confidence indication.

        Args:
            query: Original query.
            context: Retrieved context.
            confidence: Confidence score.
            query_type: Type of query.

        Returns:
            Dict with 'answer' and optionally 'disclaimer'.
        """
        answer = self.generate(query, context, query_type)

        result = {'answer': answer, 'disclaimer': None}

        if confidence < self.CONFIDENCE_DISCLAIMER_THRESHOLD:
            result['disclaimer'] = (
                "Note: This answer is based on limited matching content "
                "and may not fully address your question."
            )

        return result

    def generate_with_sources(self, query: str, context: str,
                               sources: List[Dict[str, str]],
                               query_type: str = 'default') -> str:
        """Generate answer with source citations.

        Args:
            query: Original query.
            context: Retrieved context.
            sources: List of source dicts.
            query_type: Type of query.

        Returns:
            Answer with sources appended.
        """
        raw_answer = self.generate(query, context, query_type)
        return format_answer(raw_answer, sources=sources)
