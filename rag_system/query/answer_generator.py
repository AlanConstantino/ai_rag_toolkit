"""Answer generator module for the RAG system.

Generates answers from context using LLM with grounding safeguards.
"""

import re
from typing import Dict, List, Any, Optional

from rag_system.utils import get_logger
from rag_system.query.citation_validator import extract_citations, validate_citations

logger = get_logger(__name__)


# Grounded answer prompt template with citation requirements
GROUNDED_PROMPT_TEMPLATE = """You are a documentation assistant. Answer the question using ONLY the provided context.

CRITICAL INSTRUCTIONS:
1. Your answer must come from the context below - do not use outside knowledge
2. Cite your sources using [CHUNK:id] format for each fact you state
3. If you cannot answer from the context, say so honestly
4. Do not fabricate or hallucinate information

CONTEXT (with chunk IDs for citation):
{context}

QUESTION: {query}

Provide a clear, accurate answer with [CHUNK:id] citations for each claim. If the context doesn't contain enough information to answer, explain what's missing."""


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


def build_grounded_answer_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Build prompt for grounded answer generation with chunk IDs.

    Args:
        query: Original query.
        chunks: List of chunk dicts with 'id' and 'content' keys.

    Returns:
        Complete prompt string with chunk IDs for citation.
    """
    if not chunks:
        context = "No context available."
    else:
        context_parts = []
        for chunk in chunks:
            chunk_id = chunk.get('id', 0)
            content = chunk.get('content', '')
            context_parts.append(f"[CHUNK:{chunk_id}]\n{content}")
        context = "\n\n".join(context_parts)

    return GROUNDED_PROMPT_TEMPLATE.format(query=query, context=context)


def format_fallback_response(chunks: List[Dict[str, Any]]) -> str:
    """Format a fallback response with relevant passages.

    Instead of saying "I don't know", shows the user what was found
    to keep them moving forward.

    Args:
        chunks: List of chunk dicts with content and source info.

    Returns:
        Constructive fallback response with relevant passages.
    """
    if not chunks:
        return (
            "I couldn't find relevant information to answer your question. "
            "Try rephrasing with different terms or checking the documentation directly."
        )

    response_parts = [
        "I couldn't find an exact answer to your question, but here are "
        "the most relevant passages I found:\n"
    ]

    for chunk in chunks[:3]:  # Limit to top 3 chunks
        chunk_id = chunk.get('id', 0)
        content = chunk.get('content', '')
        source = chunk.get('page_url', '') or chunk.get('page_title', 'Unknown source')

        # Truncate content if too long
        if len(content) > 200:
            content = content[:200] + "..."

        response_parts.append(f'\n[CHUNK:{chunk_id}] "{content}"')
        response_parts.append(f"Source: {source}\n")

    response_parts.append("\nYou may find your answer in these sections.")

    return "".join(response_parts)


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

    def generate_grounded(self, query: str, chunks: List[Dict[str, Any]],
                          confidence: float = 1.0) -> Dict[str, Any]:
        """Generate a grounded answer with citation validation.

        Uses stronger grounding prompts and validates that all citations
        reference real chunks to catch hallucinations.

        Args:
            query: Original query.
            chunks: List of chunk dicts with 'id', 'content', and source info.
            confidence: Confidence score (0-1). Low confidence triggers fallback.

        Returns:
            Dict with:
                - answer: The generated answer text
                - citations_valid: True if all citations reference real chunks
                - valid_citations: List of valid chunk IDs cited
                - invalid_citations: List of fabricated chunk IDs
                - used_fallback: True if fallback response was used
        """
        # Build set of valid chunk IDs
        valid_chunk_ids = {chunk.get('id') for chunk in chunks if chunk.get('id')}

        # Use fallback for very low confidence
        if confidence < 0.3 or not chunks:
            return {
                'answer': format_fallback_response(chunks),
                'citations_valid': True,
                'valid_citations': [],
                'invalid_citations': [],
                'used_fallback': True
            }

        if not self.chat_client:
            return {
                'answer': "Answer generation is not configured.",
                'citations_valid': True,
                'valid_citations': [],
                'invalid_citations': [],
                'used_fallback': False
            }

        try:
            # Build grounded prompt with chunk IDs
            prompt = build_grounded_answer_prompt(query, chunks)
            raw_answer = self.chat_client.complete(prompt)

            # Validate citations in the response
            validation = validate_citations(raw_answer, valid_chunk_ids)

            return {
                'answer': raw_answer.strip(),
                'citations_valid': validation.is_valid,
                'valid_citations': validation.valid_citations,
                'invalid_citations': validation.invalid_citations,
                'used_fallback': False
            }

        except Exception as e:
            logger.error(f"Grounded answer generation failed: {e}")
            return {
                'answer': format_fallback_response(chunks),
                'citations_valid': True,
                'valid_citations': [],
                'invalid_citations': [],
                'used_fallback': True
            }
