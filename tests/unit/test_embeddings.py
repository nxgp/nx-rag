"""
Unit tests for Milestone 3 (M3) Embedding Providers.

Tests base interface, Bedrock provider mocking, local provider formatting,
disk caching, and provider factory instantiation.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mentera_rag.embeddings.bedrock import BedrockEmbeddingProvider
from mentera_rag.embeddings.cache import EmbeddingCache
from mentera_rag.embeddings.factory import EmbeddingFactory


class TestEmbeddingCache:
    """Tests for file-based JSON embedding cache."""

    def test_cache_get_and_set(self, tmp_path: Path):
        cache = EmbeddingCache(cache_dir=tmp_path)
        model = "test-model"
        text = "Hello world"
        dim = 4
        vector = [0.1, 0.2, 0.3, 0.4]

        # Cache miss initially
        assert cache.get(model, text, dim) is None

        # Store in cache
        cache.set(model, text, dim, vector)

        # Cache hit
        cached_vector = cache.get(model, text, dim)
        assert cached_vector == vector


class TestBedrockEmbeddingProvider:
    """Tests for BedrockEmbeddingProvider boto3 invocation logic."""

    @patch("boto3.client")
    def test_bedrock_titan_embedding(self, mock_boto_client):
        mock_bedrock = MagicMock()
        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode())
        }
        mock_bedrock.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_bedrock

        provider = BedrockEmbeddingProvider(model_name="amazon.titan-embed-text-v2:0", dimension=3)

        vec = provider.embed_query("Medical text")
        assert vec == [0.1, 0.2, 0.3]
        mock_bedrock.invoke_model.assert_called_once()


class TestEmbeddingFactory:
    """Tests for EmbeddingFactory provider creation."""

    @patch("boto3.client")
    def test_get_bedrock_provider(self, mock_boto):
        provider = EmbeddingFactory.get_provider(provider_type="bedrock", model_name="titan")
        assert isinstance(provider, BedrockEmbeddingProvider)

    @patch("mentera_rag.embeddings.azure_openai.AzureOpenAI")
    def test_get_azure_provider(self, mock_azure):
        from mentera_rag.embeddings.azure_openai import AzureOpenAIEmbeddingProvider

        provider = EmbeddingFactory.get_provider(
            provider_type="azure", model_name="text-embedding-3-small"
        )
        assert isinstance(provider, AzureOpenAIEmbeddingProvider)

    @patch("vertexai.init")
    @patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
    def test_get_gcp_provider(self, mock_from_pretrained, mock_vertex_init):
        from mentera_rag.embeddings.gcp_vertex import GCPVertexEmbeddingProvider

        provider = EmbeddingFactory.get_provider(
            provider_type="gcp", model_name="text-embedding-005"
        )
        assert provider.model is not None
        assert isinstance(provider, GCPVertexEmbeddingProvider)

    def test_unsupported_provider_raises_error(self):
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            EmbeddingFactory.get_provider(provider_type="unknown_provider")
