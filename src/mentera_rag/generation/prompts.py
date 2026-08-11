"""
Domain-agnostic prompt templates for agentic retrieval graph.

These prompts are used only by the query rewriter and relevance grader nodes
in the AgenticRAGGraph. The RAG API itself does NOT generate answers —
it serves retrieved context to downstream Agent/LLM applications.
"""

RELEVANCE_GRADER_SYSTEM_PROMPT = """You are a relevance evaluator assessing whether a retrieved
document passage is relevant to a user query.
Respond with ONLY a JSON object with a single boolean field "relevant": true or false.
"""

RELEVANCE_GRADER_USER_PROMPT = """Retrieved Document:
{document}

User Query: {query}

JSON Response:"""


QUERY_REWRITER_SYSTEM_PROMPT = """You are a query optimization assistant.
Your goal is to rewrite the input user query to make it more specific and optimized
for semantic document retrieval.
Return ONLY the rewritten search query text without explanations.
"""

QUERY_REWRITER_USER_PROMPT = """Original User Query: {query}

Optimized Search Query:"""
