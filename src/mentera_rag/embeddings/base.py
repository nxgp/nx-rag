"""
Abstract Base Interface for Embedding Providers.
Establishes the contract for translating text into dense vector arrays.
Decouples vector store indexing and retrieval strategies from specific provider APIs.
"""

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """
    Abstract Base Class for all embedding models (AWS Bedrock and Local open0source).
    """

    def __init__(self, model_name: str, dimension: int):
        """
        Initialize base embedding provider properties.

        Args:
            model_name: Identifier for the model (e.g. 'amazon.titan-embed-text-v2:0').
            dimension: Target vector dimensionality (e.g. 1024, 512, 256).
        """
        self.model_name = model_name
        self.dimension = dimension

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of document/passage texts for vector store indexing.

        Applies document-specific prefixes (e.g. 'passage: ' for E5 or
        input_type='search_document' for Cohere v3).

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vectors, each of length `self.dimension`.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single search query on the hot query execution path.

        Applies query-specific prefixes (e.g. 'query: ' for E5 or
        input_type='search_query' for Cohere v3).

        Args:
            text: Query text string.

        Returns:
            A single float vector of length `self.dimension`.
        """
        pass
