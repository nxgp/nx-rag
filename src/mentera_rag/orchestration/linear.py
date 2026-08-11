"""
Linear RAG Pipeline Implementation — Mentera RAG Pipeline.

Baseline control flow executing Retrieve -> Rerank sequence.
Returns retrieved contexts directly without LLM answer generation.
"""

from typing import Any

from mentera_rag.retrieval.base import BaseReranker, BaseRetriever


class LinearRAGPipeline:
    """
    Linear RAG pipeline for baseline context retrieval.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_provider: Any = None,  # Kept for signature compatibility
        reranker: BaseReranker | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker

    def run(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute linear Retrieve -> Rerank pipeline."""
        # 1. Retrieve candidates with filters
        candidate_chunks = self.retriever.retrieve(query, top_k=top_k, filters=filters)

        # 2. Rerank if reranker is provided
        if self.reranker and candidate_chunks:
            final_chunks = self.reranker.rerank(query, candidate_chunks, top_n=top_n)
        else:
            final_chunks = candidate_chunks[:top_n]

        return {
            "query": query,
            "answer": "",  # Context-only pipeline
            "retrieved_chunks": final_chunks,
            "pipeline_type": "linear",
        }
