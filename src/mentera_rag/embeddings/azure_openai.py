"""
Azure OpenAI Embedding Provider Implementation.

Uses the official openai Python SDK (v1.0.0+) to generate dense vectors
using Azure OpenAI Service embedding deployments.
"""

import logging
from typing import Any

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from mentera_rag.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class AzureOpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider for Azure OpenAI Service.
    """

    def __init__(
        self,
        deployment_name: str = "text-embedding-3-large",
        model_name: str = "text-embedding-3-large",
        dimension: int = 1024,
        endpoint: str = "",
        api_key: str = "",
        api_version: str = "2024-02-01",
        batch_size: int = 64,
    ):
        """
        Initialize the Azure OpenAI Embedding provider.

        Args:
            deployment_name: Name of the Azure deployment.
            model_name: Name of the base model (e.g. text-embedding-3-large).
            dimension: Target vector dimensionality (e.g. 1024, 1536).
            endpoint: Azure endpoint URL (e.g. https://<resource>.openai.azure.com/).
            api_key: Azure OpenAI API Key.
            api_version: Azure OpenAI API version.
            batch_size: Batch size for document embedding.
        """
        super().__init__(model_name=model_name, dimension=dimension)
        self.deployment_name = deployment_name
        self.batch_size = batch_size

        if not endpoint or not api_key:
            logger.warning(
                "Azure OpenAI endpoint or API key not provided. "
                "API calls will fail unless configured via env variables."
            )

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _create_embeddings(self, **kwargs: Any) -> Any:
        return self.client.embeddings.create(**kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents in batches using Azure OpenAI.
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []
        is_ada = "ada" in self.model_name.lower() or "ada" in self.deployment_name.lower()

        # Batch process text items
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            kwargs: dict[str, Any] = {
                "input": batch,
                "model": self.deployment_name,
            }
            # Only pass dimensions if not using older models like text-embedding-ada-002
            if not is_ada and self.dimension is not None:
                kwargs["dimensions"] = self.dimension

            response = self._create_embeddings(**kwargs)
            embeddings.extend([list(data.embedding) for data in response.data])

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.
        """
        results = self.embed_documents([text])
        if not results:
            raise ValueError("Failed to generate embedding for the query.")
        return results[0]
