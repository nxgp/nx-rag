import sys
import pytest
from unittest.mock import MagicMock, patch
from mentera_rag.embeddings.gcp_vertex import GCPVertexEmbeddingProvider


@pytest.mark.unit
def test_gcp_embeddings_init():
    """Test initializing the GCP Vertex AI embedding provider."""
    provider = GCPVertexEmbeddingProvider(
        model_name="text-embedding-005",
        dimension=256,
        project_id="my-project",
        location="us-east4",
    )
    assert provider.model_name == "text-embedding-005"
    assert provider.dimension == 256
    assert provider.project_id == "my-project"
    assert provider.location == "us-east4"


@pytest.mark.unit
@patch("vertexai.init")
@patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
def test_gcp_embed_documents(mock_from_pretrained, mock_vertex_init):
    """Test embedding documents with mocked Vertex AI SDK."""
    mock_model_instance = MagicMock()
    mock_from_pretrained.return_value = mock_model_instance

    # Mock response format: list of objects with a .values property
    mock_val1 = MagicMock()
    mock_val1.values = [0.1, 0.2]
    mock_val2 = MagicMock()
    mock_val2.values = [0.3, 0.4]
    mock_model_instance.get_embeddings.return_value = [mock_val1, mock_val2]

    provider = GCPVertexEmbeddingProvider(
        model_name="text-embedding-005",
        dimension=2,
        project_id="my-project",
        location="us-central1",
    )

    texts = ["hello", "world"]
    vectors = provider.embed_documents(texts)

    # Verify lazy init called vertexai.init and from_pretrained
    mock_vertex_init.assert_called_once_with(project="my-project", location="us-central1")
    mock_from_pretrained.assert_called_once_with("text-embedding-005")

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2]
    assert vectors[1] == [0.3, 0.4]

    # Verify task_type and dimensions parameters passed to get_embeddings
    mock_model_instance.get_embeddings.assert_called_once()
    args, kwargs = mock_model_instance.get_embeddings.call_args
    assert "inputs" in kwargs
    assert len(kwargs["inputs"]) == 2
    assert kwargs["inputs"][0].text == "hello"
    assert kwargs["inputs"][0].task_type == "RETRIEVAL_DOCUMENT"
    assert kwargs["output_dimensionality"] == 2


@pytest.mark.unit
@patch("vertexai.init")
@patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
def test_gcp_embed_query(mock_from_pretrained, mock_vertex_init):
    """Test embedding query with mocked Vertex AI SDK."""
    mock_model_instance = MagicMock()
    mock_from_pretrained.return_value = mock_model_instance

    mock_val = MagicMock()
    mock_val.values = [0.8, 0.9]
    mock_model_instance.get_embeddings.return_value = [mock_val]

    provider = GCPVertexEmbeddingProvider(
        model_name="text-embedding-005",
        dimension=2,
        project_id="my-project",
    )

    query_vector = provider.embed_query("search query")
    assert query_vector == [0.8, 0.9]

    # Verify task_type="RETRIEVAL_QUERY" was used
    mock_model_instance.get_embeddings.assert_called_once()
    args, kwargs = mock_model_instance.get_embeddings.call_args
    assert kwargs["inputs"][0].text == "search query"
    assert kwargs["inputs"][0].task_type == "RETRIEVAL_QUERY"
