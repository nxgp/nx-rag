"""
Embedding Provider Factory — Mentera RAG Pipeline.

Supports three cloud embedding providers:
  - bedrock : AWS Bedrock (Titan v2, Cohere v3)
  - azure   : Azure OpenAI (text-embedding-3-small/large, ada-002)
  - gcp     : GCP Vertex AI (text-embedding-005, text-multilingual-embedding-002)

Local/open-source models are not supported in this pipeline.
"""

from mentera_rag.config.settings import settings
from mentera_rag.embeddings.base import BaseEmbeddingProvider
from mentera_rag.embeddings.bedrock import BedrockEmbeddingProvider


class EmbeddingFactory:
    """
    Factory class creating cloud embedding provider instances from configuration.
    """

    @staticmethod
    def get_provider(
        provider_type: str | None = None,
        model_name: str | None = None,
        dimension: int | None = None,
    ) -> BaseEmbeddingProvider:
        """
        Create and return a cloud embedding provider instance.

        Args:
            provider_type: 'bedrock', 'azure', or 'gcp'.
                           Defaults to settings.DEFAULT_EMBEDDING_PROVIDER.
            model_name: Provider-specific model ID.
                        Defaults to settings.DEFAULT_EMBEDDING_MODEL.
            dimension: Embedding vector dimensionality.
                       Defaults to settings.DEFAULT_EMBEDDING_DIMENSION.

        Returns:
            Configured BaseEmbeddingProvider instance.

        Raises:
            ValueError: If provider_type is not one of the supported values.
            ImportError: If Azure/GCP SDK is not installed.
        """
        p_type = (provider_type or settings.DEFAULT_EMBEDDING_PROVIDER).lower()
        m_name = model_name or settings.DEFAULT_EMBEDDING_MODEL
        dim = dimension or settings.DEFAULT_EMBEDDING_DIMENSION

        if p_type == "bedrock":
            return BedrockEmbeddingProvider(
                model_name=m_name,
                dimension=dim,
                region_name=settings.AWS_REGION,
            )

        if p_type == "azure":
            # Lazy import — requires: pip install openai
            try:
                from mentera_rag.embeddings.azure_openai import AzureOpenAIEmbeddingProvider
            except ImportError as e:
                raise ImportError(
                    "Azure OpenAI embedding requires the openai package. "
                    "Install with: pip install mentera-rag"
                ) from e
            return AzureOpenAIEmbeddingProvider(
                deployment_name=settings.AZURE_EMBEDDING_DEPLOYMENT,
                model_name=m_name,
                dimension=dim,
                endpoint=settings.AZURE_OPENAI_ENDPOINT or "",
                api_key=settings.AZURE_OPENAI_API_KEY or "",
                api_version=settings.AZURE_OPENAI_API_VERSION,
            )

        if p_type == "gcp":
            # Lazy import — requires: pip install google-cloud-aiplatform
            try:
                from mentera_rag.embeddings.gcp_vertex import GCPVertexEmbeddingProvider
            except ImportError as e:
                raise ImportError(
                    "GCP Vertex AI embedding requires google-cloud-aiplatform. "
                    "Install with: pip install mentera-rag"
                ) from e
            return GCPVertexEmbeddingProvider(
                model_name=m_name or settings.GCP_EMBEDDING_MODEL,
                dimension=dim,
                project_id=settings.GCP_PROJECT_ID or "",
                location=settings.GCP_LOCATION,
            )

        raise ValueError(
            f"Unsupported embedding provider: '{p_type}'. "
            "Choose one of: 'bedrock', 'azure', 'gcp'."
        )
