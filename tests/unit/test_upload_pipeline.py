import pytest
import hashlib
from unittest.mock import MagicMock, patch
from mentera_rag.ingestion.upload_pipeline import UploadPipeline


@pytest.fixture
def mock_pipeline_components():
    """Mock storage, embedding, vector store, and parser components."""
    with patch("mentera_rag.ingestion.upload_pipeline.StorageFactory.get_store") as mock_get_store, \
         patch("mentera_rag.ingestion.upload_pipeline.EmbeddingFactory.get_provider") as mock_get_embed, \
         patch("mentera_rag.ingestion.upload_pipeline.VectorStoreFactory.get_vector_store") as mock_get_qdrant, \
         patch("mentera_rag.ingestion.upload_pipeline.ParserFactory.get_parser") as mock_get_parser, \
         patch("mentera_rag.ingestion.upload_pipeline.ParserFactory.is_supported") as mock_is_supported:

        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        mock_embed = MagicMock()
        mock_embed.model_name = "test-embedding-model"
        mock_get_embed.return_value = mock_embed

        mock_qdrant = MagicMock()
        mock_get_qdrant.return_value = mock_qdrant

        mock_parser = MagicMock()
        mock_get_parser.return_value = mock_parser

        mock_is_supported.return_value = True

        yield {
            "store": mock_store,
            "embed": mock_embed,
            "qdrant": mock_qdrant,
            "parser": mock_parser,
            "is_supported": mock_is_supported,
        }


@pytest.mark.unit
def test_upload_pipeline_success(mock_pipeline_components):
    """Test successful run of the ingestion pipeline."""
    components = mock_pipeline_components

    # Mock download bytes
    dummy_bytes = b"Hello from S3 object content text to split."
    components["store"].download.return_value = dummy_bytes

    # Mock Qdrant deduplication check to return empty (not already indexed)
    components["qdrant"].client.scroll.return_value = ([], None)

    # Mock parser output: one page
    from mentera_rag.parsing.base import ParsedPage
    components["parser"].parse.return_value = [
        ParsedPage(content="Parsed document page text.", page_number=1)
    ]

    # Mock embedding output
    components["embed"].embed_documents.return_value = [[0.1, 0.2]]

    # Run pipeline
    pipeline = UploadPipeline(
        storage_provider="local",
        embedding_provider="bedrock",
    )
    result = pipeline.run(
        storage_key="tenant123/provider456/uuid_file.txt",
        tenant_id="tenant123",
        provider_id="provider456",
        tags=["general"],
    )

    assert result["status"] == "success"
    assert result["chunk_count"] == 1
    assert result["embedding_model"] == "test-embedding-model"
    assert result["document_id"].startswith("doc_tenant123_")

    # Verify download, parse, embed and index were called
    components["store"].download.assert_called_once_with("tenant123/provider456/uuid_file.txt")
    components["parser"].parse.assert_called_once()
    components["embed"].embed_documents.assert_called_once_with(["Parsed document page text."])
    components["qdrant"].index_chunks.assert_called_once()


@pytest.mark.unit
def test_upload_pipeline_deduplication(mock_pipeline_components):
    """Test that duplicate files are skipped during ingestion using the deduplication check."""
    components = mock_pipeline_components

    dummy_bytes = b"Duplicate file content."
    components["store"].download.return_value = dummy_bytes

    # Mock scroll result to return an existing chunk (indicating file hash exists for this tenant)
    mock_point = MagicMock()
    mock_point.payload = {
        "doc_id": "doc_tenant123_existing_doc",
        "embedding_model": "existing-model",
    }
    components["qdrant"].client.scroll.return_value = ([mock_point], None)

    pipeline = UploadPipeline()
    result = pipeline.run(
        storage_key="tenant123/provider456/uuid_duplicate.txt",
        tenant_id="tenant123",
        provider_id="provider456",
    )

    # Ingestion skipped: chunk count is 0 and status is already_indexed
    assert result["status"] == "already_indexed"
    assert result["chunk_count"] == 0
    assert result["document_id"] == "doc_tenant123_existing_doc"

    # Ingestion steps (parse, embed, index) should NOT be called
    components["parser"].parse.assert_not_called()
    components["embed"].embed_documents.assert_not_called()
    components["qdrant"].index_chunks.assert_not_called()


@pytest.mark.unit
def test_upload_pipeline_unsupported_format(mock_pipeline_components):
    """Test that unsupported file extensions raise a ValueError."""
    components = mock_pipeline_components
    components["is_supported"].return_value = False

    pipeline = UploadPipeline()
    with pytest.raises(ValueError, match="Unsupported file extension"):
        pipeline.run(
            storage_key="tenant123/provider456/file.exe",
            tenant_id="tenant123",
            provider_id="provider456",
        )
