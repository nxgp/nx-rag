"""
Unit tests for Qdrant Vector Store Adapter and Factory.
"""

from unittest.mock import MagicMock, patch
import pytest

from mentera_rag.chunking.schemas import Chunk
from mentera_rag.vector_stores.factory import VectorStoreFactory
from mentera_rag.vector_stores.qdrant_store import QdrantVectorStore
from mentera_rag.vector_stores.schemas import VectorSearchResult


class TestVectorStoreSchemas:
    """Tests for VectorSearchResult schema."""

    def test_vector_search_result_instantiation(self):
        res = VectorSearchResult(
            chunk_id="chunk_101",
            doc_id="doc_1",
            text="Heart failure treatment involves ACE inhibitors.",
            score=0.92,
            tenant_id="tenant123",
            provider_id="provider456",
            metadata={"source": "pubmed"},
        )
        assert res.chunk_id == "chunk_101"
        assert res.doc_id == "doc_1"
        assert res.score == 0.92
        assert res.metadata["source"] == "pubmed"


class TestQdrantVectorStore:
    """Tests for QdrantVectorStore adapter."""

    @patch("mentera_rag.vector_stores.qdrant_store.QdrantClient")
    def test_create_collection(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_qdrant_class.return_value = mock_client

        store = QdrantVectorStore(collection_name="test_col", dimension=512)
        store.create_collection()

        mock_client.collection_exists.assert_called_with("test_col")
        mock_client.create_collection.assert_called_once()

    @patch("mentera_rag.vector_stores.qdrant_store.QdrantClient")
    def test_create_collection_force_recreate(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_qdrant_class.return_value = mock_client

        store = QdrantVectorStore(collection_name="test_col", dimension=512)
        store.create_collection(force_recreate=True)

        mock_client.delete_collection.assert_called_with("test_col")
        mock_client.create_collection.assert_called_once()

    @patch("mentera_rag.vector_stores.qdrant_store.QdrantClient")
    def test_index_chunks(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client

        store = QdrantVectorStore(collection_name="test_col", dimension=2)
        chunks = [
            Chunk(
                chunk_id="c1",
                doc_id="d1",
                chunk_index=0,
                text="Chunk text 1",
                token_count=4,
                tenant_id="tenant123",
                provider_id="provider456",
            ),
            Chunk(
                chunk_id="c2",
                doc_id="d1",
                chunk_index=1,
                text="Chunk text 2",
                token_count=4,
                tenant_id="tenant123",
                provider_id="provider456",
            ),
        ]
        vectors = [[0.1, 0.2], [0.3, 0.4]]

        store.index_chunks(chunks, vectors)
        mock_client.upsert.assert_called_once()

    @patch("mentera_rag.vector_stores.qdrant_store.QdrantClient")
    def test_index_chunks_mismatched_length_raises_error(self, mock_qdrant_class):
        store = QdrantVectorStore(collection_name="test_col", dimension=2)
        chunks = [
            Chunk(
                chunk_id="c1",
                doc_id="d1",
                chunk_index=0,
                text="Text",
                token_count=1,
                tenant_id="tenant123",
                provider_id="provider456",
            )
        ]
        vectors = [[0.1, 0.2], [0.3, 0.4]]

        with pytest.raises(ValueError, match="count must match"):
            store.index_chunks(chunks, vectors)

    @patch("mentera_rag.vector_stores.qdrant_store.QdrantClient")
    def test_search_dense(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_point = MagicMock()
        mock_point.id = "c1"
        mock_point.score = 0.88
        mock_point.payload = {
            "chunk_id": "c1",
            "doc_id": "d1",
            "text": "Heart disease context",
            "tenant_id": "tenant123",
            "provider_id": "provider456",
        }

        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_client.query_points.return_value = mock_response
        mock_qdrant_class.return_value = mock_client

        store = QdrantVectorStore(collection_name="test_col", dimension=2)
        results = store.search_dense(query_vector=[0.1, 0.2], top_k=1)

        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].score == 0.88

    @patch("mentera_rag.vector_stores.qdrant_store.QdrantClient")
    def test_search_hybrid(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_point1 = MagicMock()
        mock_point1.id = "c1"
        mock_point1.score = 0.88
        mock_point1.payload = {
            "chunk_id": "c1",
            "doc_id": "d1",
            "text": "Asthma inhalers treatment",
            "tenant_id": "tenant123",
            "provider_id": "provider456",
        }

        mock_point2 = MagicMock()
        mock_point2.id = "c2"
        mock_point2.score = 0.65
        mock_point2.payload = {
            "chunk_id": "c2",
            "doc_id": "d2",
            "text": "Diabetes insulin therapy",
            "tenant_id": "tenant123",
            "provider_id": "provider456",
        }

        mock_response = MagicMock()
        mock_response.points = [mock_point1, mock_point2]
        mock_client.query_points.return_value = mock_response
        mock_qdrant_class.return_value = mock_client

        store = QdrantVectorStore(collection_name="test_col", dimension=2)
        results = store.search_hybrid(
            query_text="asthma inhalers", query_vector=[0.1, 0.2], top_k=2
        )

        assert len(results) == 2
        assert results[0].chunk_id == "c1"

    @patch("mentera_rag.vector_stores.qdrant_store.QdrantClient")
    def test_delete_collection(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_qdrant_class.return_value = mock_client

        store = QdrantVectorStore(collection_name="test_col", dimension=2)
        store.delete_collection()

        mock_client.delete_collection.assert_called_with("test_col")


class TestVectorStoreFactory:
    """Tests for VectorStoreFactory creation."""

    @patch("mentera_rag.vector_stores.qdrant_store.QdrantClient")
    def test_get_qdrant_store(self, mock_qdrant):
        store = VectorStoreFactory.get_vector_store()
        assert isinstance(store, QdrantVectorStore)
