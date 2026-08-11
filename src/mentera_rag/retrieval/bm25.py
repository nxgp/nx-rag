"""
Sparse BM25 Retriever Implementation (M5).

Uses BM25Okapi term frequency-inverse document frequency matching over text chunks.
"""

from rank_bm25 import BM25Okapi

from mentera_rag.chunking.schemas import Chunk
from mentera_rag.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """
    Sparse keyword retriever using BM25Okapi algorithm.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

        # Tokenize corpus text into lowercased word tokens
        corpus_tokens = [c.text.lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str, top_k: int = 10, filters: dict | None = None) -> list[Chunk]:
        """Tokenize query and return top-k BM25 keyword matches, applying filters if provided."""
        filtered_chunks = self.chunks
        if filters:
            filtered_chunks = [
                c for c in self.chunks
                if all(
                    (getattr(c, k, None) == v or c.metadata.get(k) == v)
                    for k, v in filters.items()
                    if v is not None
                )
            ]

        if not filtered_chunks:
            return []

        # Tokenize only the filtered corpus chunks
        corpus_tokens = [c.text.lower().split() for c in filtered_chunks]
        bm25 = BM25Okapi(corpus_tokens)

        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        # Pair chunks with scores and sort in descending order
        scored_chunks = list(zip(filtered_chunks, scores, strict=False))
        scored_chunks.sort(key=lambda x: float(x[1]), reverse=True)

        return [chunk for chunk, score in scored_chunks[:top_k]]
