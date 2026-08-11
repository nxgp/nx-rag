"""
Factory for Instantiating Vector Stores.
Mentera RAG uses Qdrant exclusively — shared collection with tenant_id filtering.
"""

from mentera_rag.config.settings import settings
from mentera_rag.vector_stores.base import BaseVectorStore
from mentera_rag.vector_stores.qdrant_store import QdrantVectorStore


class VectorStoreFactory:
    """
    Factory class creating Qdrant vector store instances.
    """

    @staticmethod
    def get_vector_store(
        collection_name: str = "mentera_chunks",
        dimension: int = 1024,
    ) -> BaseVectorStore:
        """
        Creates and returns a QdrantVectorStore instance.

        Args:
            collection_name: Name of the Qdrant collection (shared across all tenants).
            dimension: Embedding vector dimensionality.

        Returns:
            Configured QdrantVectorStore instance.
        """
        return QdrantVectorStore(
            collection_name=collection_name,
            dimension=dimension,
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
