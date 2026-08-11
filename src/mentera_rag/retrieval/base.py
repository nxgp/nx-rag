"""
Abstract Interfaces for Retrievers and Rerankers (M5).

Establishes standard contracts for retrieving chunks and re-scoring them.
"""

from abc import ABC, abstractmethod

from mentera_rag.chunking.schemas import Chunk


class BaseRetriever(ABC):
    """
    Abstract Base Class for all retrieval strategies (Dense, BM25, Ensemble/Fusion).
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10, filters: dict | None = None) -> list[Chunk]:
        """
        Retrieve top-k relevant text chunks for a query string.

        Args:
            query: User search question string.
            top_k: Number of candidate chunks to retrieve.
            filters: Optional tenant isolation and metadata filter dictionary.

        Returns:
            List of Chunk objects sorted by relevance score.
        """
        pass


class BaseReranker(ABC):
    """
    Abstract Base Class for candidate reranking models (Bedrock Rerank & Local Cross-Encoders).
    """

    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], top_n: int = 5) -> list[Chunk]:
        """
        Re-score and trim candidate chunks using a Cross-Encoder or Reranking API.

        Args:
            query: User search question string.
            chunks: List of candidate Chunk objects from retriever.
            top_n: Number of final top chunks to return.

        Returns:
            List of top-n re-ranked Chunk objects.
        """
        pass
