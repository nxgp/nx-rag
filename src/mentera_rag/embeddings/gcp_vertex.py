"""
GCP Vertex AI Embedding Provider Implementation.

Uses the google-cloud-aiplatform SDK to generate dense vectors
using Vertex AI's text-embedding models (e.g. text-embedding-005).
"""

import logging
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from mentera_rag.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class GCPVertexEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider for GCP Vertex AI text-embedding models.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-005",
        dimension: int = 1024,
        project_id: str = "",
        location: str = "us-central1",
        batch_size: int = 250,  # Vertex AI supports up to 250 instances per request
    ):
        """
        Initialize the GCP Vertex AI Embedding provider.

        Args:
            model_name: Name of the Vertex model (e.g. text-embedding-005).
            dimension: Target vector dimensionality (e.g. 1024, 768, 512, 256, 128).
            project_id: GCP project ID.
            location: GCP region (e.g. us-central1).
            batch_size: Batch size for document embedding (max 250).
        """
        super().__init__(model_name=model_name, dimension=dimension)
        self.project_id = project_id
        self.location = location
        self.batch_size = min(batch_size, 250)
        self._model: Any = None

    @property
    def model(self) -> Any:
        """
        Lazy-initialize the Vertex AI SDK and TextEmbeddingModel.
        """
        if self._model is None:
            try:
                import vertexai
                from vertexai.language_models import TextEmbeddingModel

                vertexai.init(project=self.project_id, location=self.location)
                self._model = TextEmbeddingModel.from_pretrained(self.model_name)
            except Exception as e:
                logger.error("Failed to initialize GCP Vertex AI TextEmbeddingModel: %s", e)
                raise RuntimeError(
                    "GCP Vertex AI SDK initialization failed. Ensure credentials are configured."
                ) from e
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _get_embeddings(self, **kwargs: Any) -> Any:
        return self.model.get_embeddings(**kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents in batches using GCP Vertex AI.
        """
        if not texts:
            return []

        from vertexai.language_models import TextEmbeddingInput

        embeddings: list[list[float]] = []

        # Vertex AI supports up to 250 texts per batch
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # task_type="RETRIEVAL_DOCUMENT" is optimal for documents to be indexed
            inputs = [TextEmbeddingInput(text, task_type="RETRIEVAL_DOCUMENT") for text in batch]

            kwargs: dict[str, Any] = {"inputs": inputs}
            # text-embedding-004 and text-embedding-005 support output_dimensionality
            if "text-embedding-00" in self.model_name:
                kwargs["output_dimensionality"] = self.dimension

            response = self._get_embeddings(**kwargs)
            embeddings.extend([list(emb.values) for emb in response])

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string using GCP Vertex AI.
        """
        from vertexai.language_models import TextEmbeddingInput

        # task_type="RETRIEVAL_QUERY" is optimal for search queries
        inputs = [TextEmbeddingInput(text, task_type="RETRIEVAL_QUERY")]

        kwargs: dict[str, Any] = {"inputs": inputs}
        if "text-embedding-00" in self.model_name:
            kwargs["output_dimensionality"] = self.dimension

        response = self._get_embeddings(**kwargs)
        if not response:
            raise ValueError("Failed to generate Vertex AI embedding for the query.")

        return list(response[0].values)
