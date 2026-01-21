"""Query expander module for the RAG system.

Expands queries with synonyms and variations for better retrieval.
"""

from typing import List, Optional, Any, Set

from rag_system.utils import get_logger, tokenize

logger = get_logger(__name__)


# Synonym mappings
SYNONYMS = {
    'configure': ['setup', 'set up', 'config', 'configuration'],
    'setup': ['configure', 'set up', 'config', 'configuration'],
    'set up': ['configure', 'setup', 'config', 'install'],
    'install': ['set up', 'deploy', 'installation'],
    'error': ['issue', 'problem', 'bug', 'failure'],
    'issue': ['error', 'problem', 'bug'],
    'problem': ['error', 'issue', 'bug'],
    'connect': ['connection', 'link', 'attach'],
    'connection': ['connect', 'link', 'connectivity'],
    'start': ['begin', 'launch', 'run', 'initiate'],
    'stop': ['halt', 'terminate', 'end', 'kill'],
    'create': ['make', 'build', 'generate', 'add'],
    'delete': ['remove', 'destroy', 'erase'],
    'update': ['modify', 'change', 'edit'],
    'get': ['retrieve', 'fetch', 'obtain', 'acquire'],
}

# Stopwords to remove
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
    'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'about', 'against', 'between', 'during', 'without', 'before', 'after',
    'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
    'this', 'that', 'these', 'those', 'it', 'its', 'i', 'me', 'my', 'we',
    'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
    'their', 'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too', 'very', 'just',
    'and', 'but', 'if', 'or', 'because', 'although', 'while', 'way', 'best',
}

# Expansion prompt
EXPANSION_PROMPT = """Generate 3 alternative phrasings for this search query.
Keep the same meaning but use different words.

Query: {query}

Return one alternative per line:"""


def get_synonyms(word: str) -> List[str]:
    """Get synonyms for a word.

    Args:
        word: Word to find synonyms for.

    Returns:
        List of synonyms.
    """
    return SYNONYMS.get(word.lower(), [])


def extract_key_terms(query: str) -> List[str]:
    """Extract key terms from query.

    Args:
        query: Query text.

    Returns:
        List of key terms.
    """
    tokens = tokenize(query)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def expand_with_synonyms(query: str) -> List[str]:
    """Expand query with synonym substitutions.

    Args:
        query: Query text.

    Returns:
        List of query variations.
    """
    variations = [query]
    query_lower = query.lower()

    # Try each synonym mapping
    for word, syns in SYNONYMS.items():
        if word in query_lower:
            for syn in syns[:2]:  # Limit to 2 synonyms per word
                variation = query_lower.replace(word, syn)
                if variation != query_lower and variation not in variations:
                    variations.append(variation)

    return variations


class QueryExpander:
    """Expands queries with variations for better retrieval."""

    def __init__(self, chat_client: Optional[Any] = None,
                 max_expansions: int = 5):
        """Initialize the expander.

        Args:
            chat_client: Optional chat client for LLM expansion.
            max_expansions: Maximum number of expanded queries.
        """
        self.chat_client = chat_client
        self.max_expansions = max_expansions

    def expand(self, query: str) -> List[str]:
        """Expand query using rule-based approach.

        Args:
            query: Query text.

        Returns:
            List of expanded queries.
        """
        if not query.strip():
            return [query]

        variations = [query]  # Always include original first

        # Add synonym variations
        syn_variations = expand_with_synonyms(query)
        for v in syn_variations:
            if v not in variations:
                variations.append(v)

        # Extract key terms and create term-based variation
        key_terms = extract_key_terms(query)
        if key_terms:
            term_query = ' '.join(key_terms)
            if term_query != query and term_query not in variations:
                variations.append(term_query)

        return variations[:self.max_expansions]

    def expand_with_llm(self, query: str) -> List[str]:
        """Expand query using LLM.

        Args:
            query: Query text.

        Returns:
            List of expanded queries.
        """
        variations = [query]  # Always include original

        if not self.chat_client:
            return self.expand(query)

        try:
            prompt = EXPANSION_PROMPT.format(query=query)
            result = self.chat_client.complete(prompt)

            # Parse lines from result
            for line in result.strip().split('\n'):
                line = line.strip()
                if line and line not in variations:
                    variations.append(line)

            return variations[:self.max_expansions]

        except Exception as e:
            logger.error(f"LLM expansion failed: {e}")
            return self.expand(query)
