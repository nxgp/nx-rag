"""
Abstract Base Interface for Vector Stores.

Defines the common contract for indexing and searching dense/hybrid vectors across
Qdrant and Weaviate.
"""

from abc import ABC, abstractmethod
from typing import Any

from mentera_rag.chunking.schemas import Chunk
from mentera_rag.vector_stores.schemas import VectorSearchResult


class BaseVectorStore(ABC):
    """
    Abstract Base Class for vector store adapters.
    """

    def __init__(self, collection_name: str, dimension: int = 1024):
        """
        Initialize base properties.

        Args:
            collection_name: Name of the vector collection/index.
            dimension: Dimensionality of vectors (e.g. 1024 for Titan v2 / BGE).
        """
        self.collection_name = collection_name
        self.dimension = dimension

    @abstractmethod
    def create_collection(self, force_recreate: bool = False) -> None:
        """
        Create the vector collection/index with HNSW parameters if it does not exist.

        Args:
            force_recreate: If True, drops existing collection before recreating.
        """
        pass

    @abstractmethod
    def index_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """
        Index a batch of text chunks and their corresponding dense vectors.

        Args:
            chunks: List of Chunk objects.
            vectors: List of dense vector arrays (matching chunks length).
        """
        pass

    @abstractmethod
    def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Execute pure vector (dense) similarity search.

        Args:
            query_vector: Floating-point query vector.
            top_k: Number of nearest neighbors to return.
            filters: Optional key-value dictionary for metadata filtering.

        Returns:
            List of standardized VectorSearchResult items sorted by score.
        """
        pass

    @abstractmethod
    def search_hybrid(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 10,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Execute native hybrid search combining keyword (BM25) and vector similarity.

        Args:
            query_text: Raw query string for keyword search.
            query_vector: Dense query vector for vector search.
            top_k: Number of results to return.
            alpha: Weight balancing vector search vs keyword search (0.0=BM25, 1.0=Vector).
            filters: Optional metadata filters.

        Returns:
            List of standardized VectorSearchResult items.
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Drop the collection and delete all indexed points."""
        pass
