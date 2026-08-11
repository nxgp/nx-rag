"""
Ensemble / Fusion Retriever Implementation (M5).

Combines dense vector retrieval and sparse BM25 retrieval using Reciprocal Rank Fusion (RRF).
Formula: RRF_Score(d) = sum(1 / (60 + rank(d)))
"""

from collections import defaultdict

from mentera_rag.chunking.schemas import Chunk
from mentera_rag.retrieval.base import BaseRetriever


class EnsembleRetriever(BaseRetriever):
    """
    Hybrid Ensemble Retriever using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        weights: list[float] | None = None,
        rrf_k: int = 60,
    ):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 10, filters: dict | None = None) -> list[Chunk]:
        """Execute all retrievers in parallel and combine ranks with RRF math."""
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, Chunk] = {}

        for retriever, weight in zip(self.retrievers, self.weights, strict=False):
            # Retrieve candidate list from individual retriever
            candidates = retriever.retrieve(query, top_k=top_k * 2, filters=filters)

            for rank, chunk in enumerate(candidates, start=1):
                chunk_map[chunk.chunk_id] = chunk
                # RRF Formula: weight * (1 / (rrf_k + rank))
                score = weight * (1.0 / (self.rrf_k + rank))
                rrf_scores[chunk.chunk_id] += score

        # Sort chunks by final RRF score in descending order
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        return [chunk_map[cid] for cid in sorted_ids[:top_k]]
