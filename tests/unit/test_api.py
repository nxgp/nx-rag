"""
Unit tests for the updated FastAPI REST Service endpoints.
"""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from mentera_rag.api.main import app

client = TestClient(app)


@patch("mentera_rag.vector_stores.factory.VectorStoreFactory.get_vector_store")
@patch("mentera_rag.storage.factory.StorageFactory.get_store")
def test_health_check(mock_get_storage, mock_get_qdrant):
    """Test health check endpoint with mocked connections."""
    mock_qdrant = MagicMock()
    mock_get_qdrant.return_value = mock_qdrant
    mock_qdrant.client.get_collections.return_value = []

    mock_storage = MagicMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.exists.return_value = False

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["services"]["qdrant"] == "healthy"
    assert data["services"]["storage"] == "healthy"


@patch("mentera_rag.storage.factory.StorageFactory.get_store")
def test_presign_endpoint_success(mock_get_storage):
    """Test generating a presigned URL successfully."""
    mock_storage = MagicMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.generate_presigned_upload_url.return_value = "https://mockbucket.s3.amazonaws.com/test_key?signature"

    payload = {
        "filename": "clinical_guideline.pdf",
        "content_type": "application/pdf",
        "tenant_id": "tenant123",
        "provider_id": "provider456",
    }
    response = client.post("/upload/presign", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "upload_url" in data
    assert "storage_key" in data
    assert data["expires_in"] == 3600
    assert data["storage_key"].startswith("tenant123/provider456/")


def test_presign_endpoint_invalid_tenant():
    """Test that invalid characters in tenant_id trigger a 400 validation error."""
    payload = {
        "filename": "file.txt",
        "tenant_id": "tenant/123",  # invalid slash
        "provider_id": "provider456",
    }
    response = client.post("/upload/presign", json=payload)
    assert response.status_code == 400


def test_presign_endpoint_unsupported_file():
    """Test that unsupported file extensions trigger a 400 error."""
    payload = {
        "filename": "virus.exe",
        "tenant_id": "tenant123",
        "provider_id": "provider456",
    }
    response = client.post("/upload/presign", json=payload)
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


@patch("mentera_rag.api.main.UploadPipeline")
def test_ingest_endpoint_success(mock_pipeline_class):
    """Test triggering ingestion pipeline endpoint."""
    mock_pipeline = MagicMock()
    mock_pipeline_class.return_value = mock_pipeline
    mock_pipeline.run.return_value = {
        "document_id": "doc_tenant123_abc",
        "chunk_count": 15,
        "embedding_model": "amazon.titan-embed-text-v2:0",
        "status": "success",
        "processing_time_ms": 250.5,
    }

    payload = {
        "storage_key": "tenant123/provider456/uuid_file.pdf",
        "tenant_id": "tenant123",
        "provider_id": "provider456",
        "tags": ["cardiology"],
    }
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunk_count"] == 15
    assert data["document_id"] == "doc_tenant123_abc"


@patch("mentera_rag.api.main.EmbeddingFactory.get_provider")
@patch("mentera_rag.api.main.VectorStoreFactory.get_vector_store")
def test_query_endpoint(mock_get_qdrant, mock_get_embedding):
    """Test query endpoint returns context search results with tenant filters."""
    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1, 0.2]
    mock_get_embedding.return_value = mock_embedding

    mock_qdrant = MagicMock()
    from mentera_rag.vector_stores.schemas import VectorSearchResult
    mock_qdrant.search_hybrid.return_value = [
        VectorSearchResult(
            chunk_id="chunk_1",
            doc_id="doc_1",
            text="First matched context text.",
            score=0.95,
            tenant_id="tenant123",
            provider_id="provider456",
            document_type="pdf",
            tags=["cardiology"],
            page_number=3,
        )
    ]
    mock_get_qdrant.return_value = mock_qdrant

    payload = {
        "query": "asthma treatment",
        "tenant_id": "tenant123",
        "provider_id": "provider456",
        "top_k": 5,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_results"] == 1
    assert data["retrieved_contexts"][0]["chunk_id"] == "chunk_1"
    assert data["retrieved_contexts"][0]["text"] == "First matched context text."
    assert data["retrieved_contexts"][0]["metadata"]["page_number"] == 3
