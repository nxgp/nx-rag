from unittest.mock import MagicMock, patch

import pytest

from mentera_rag.embeddings.azure_openai import AzureOpenAIEmbeddingProvider


@pytest.mark.unit
def test_azure_embeddings_init():
    """Test initializing the Azure OpenAI embedding provider."""
    provider = AzureOpenAIEmbeddingProvider(
        deployment_name="test-deployment",
        model_name="text-embedding-3-small",
        dimension=512,
        endpoint="https://test.openai.azure.com/",
        api_key="secret-key",
        api_version="2024-02-01",
    )
    assert provider.deployment_name == "test-deployment"
    assert provider.model_name == "text-embedding-3-small"
    assert provider.dimension == 512


@pytest.mark.unit
@patch("mentera_rag.embeddings.azure_openai.AzureOpenAI")
def test_azure_embed_documents(mock_azure_client):
    """Test embedding documents with mocked client."""
    mock_instance = MagicMock()
    mock_azure_client.return_value = mock_instance

    # Mock response format: response.data[i].embedding
    mock_emb1 = MagicMock()
    mock_emb1.embedding = [0.1, 0.2, 0.3]
    mock_emb2 = MagicMock()
    mock_emb2.embedding = [0.4, 0.5, 0.6]

    mock_response = MagicMock()
    mock_response.data = [mock_emb1, mock_emb2]
    mock_instance.embeddings.create.return_value = mock_response

    provider = AzureOpenAIEmbeddingProvider(
        deployment_name="test-deployment",
        model_name="text-embedding-3-small",
        dimension=3,
        endpoint="https://test.openai.azure.com/",
        api_key="secret-key",
    )

    texts = ["doc1", "doc2"]
    vectors = provider.embed_documents(texts)

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]
    assert vectors[1] == [0.4, 0.5, 0.6]

    # Verify parameters passed to the mocked openai client
    mock_instance.embeddings.create.assert_called_once_with(
        input=texts,
        model="test-deployment",
        dimensions=3,
    )


@pytest.mark.unit
@patch("mentera_rag.embeddings.azure_openai.AzureOpenAI")
def test_azure_embed_documents_ada(mock_azure_client):
    """Test that ada model does not send dimensions parameter."""
    mock_instance = MagicMock()
    mock_azure_client.return_value = mock_instance

    mock_emb = MagicMock()
    mock_emb.embedding = [0.1] * 1536
    mock_response = MagicMock()
    mock_response.data = [mock_emb]
    mock_instance.embeddings.create.return_value = mock_response

    provider = AzureOpenAIEmbeddingProvider(
        deployment_name="ada-deployment",
        model_name="text-embedding-ada-002",
        dimension=1536,
        endpoint="https://test.openai.azure.com/",
        api_key="secret-key",
    )

    provider.embed_documents(["doc"])

    # Ada-002 does not support dimensions param, verify it's omitted
    mock_instance.embeddings.create.assert_called_once_with(
        input=["doc"],
        model="ada-deployment",
    )


@pytest.mark.unit
@patch("mentera_rag.embeddings.azure_openai.AzureOpenAI")
def test_azure_embed_query(mock_azure_client):
    """Test embedding query with mocked client."""
    mock_instance = MagicMock()
    mock_azure_client.return_value = mock_instance

    mock_emb = MagicMock()
    mock_emb.embedding = [0.9, 0.8, 0.7]
    mock_response = MagicMock()
    mock_response.data = [mock_emb]
    mock_instance.embeddings.create.return_value = mock_response

    provider = AzureOpenAIEmbeddingProvider(
        deployment_name="test-deployment",
        model_name="text-embedding-3-small",
        dimension=3,
        endpoint="https://test.openai.azure.com/",
        api_key="secret-key",
    )

    query_vector = provider.embed_query("search query")
    assert query_vector == [0.9, 0.8, 0.7]
