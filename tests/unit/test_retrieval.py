"""
Unit tests for Retrievers and Rerankers.

Tests DenseRetriever, BM25Retriever, EnsembleRetriever (RRF), LocalReranker,
and BedrockReranker. All retrievers support tenant filtering.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from mentera_rag.chunking.schemas import Chunk
from mentera_rag.retrieval.bm25 import BM25Retriever
from mentera_rag.retrieval.dense import DenseRetriever
from mentera_rag.retrieval.ensemble import EnsembleRetriever
from mentera_rag.retrieval.rerankers import BedrockReranker, LocalReranker
from mentera_rag.vector_stores.schemas import VectorSearchResult


class TestDenseRetriever:
    """Tests for DenseRetriever."""

    def test_dense_retrieval_flow(self):
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1, 0.2]

        mock_store = MagicMock()
        mock_store.search_dense.return_value = [
            VectorSearchResult(
                chunk_id="c1",
                doc_id="d1",
                text="Heart treatment context",
                score=0.89,
                tenant_id="tenant123",
                provider_id="provider456",
                document_type="pdf",
                collection_name="default",
                tags=[],
                metadata={"source": "pubmed"},
            )
        ]

        retriever = DenseRetriever(embed_provider=mock_embed, vector_store=mock_store)
        results = retriever.retrieve("What is heart treatment?", top_k=1, filters={"tenant_id": "tenant123"})

        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].text == "Heart treatment context"
        assert results[0].tenant_id == "tenant123"
        mock_embed.embed_query.assert_called_once_with("What is heart treatment?")
        mock_store.search_dense.assert_called_once_with(
            query_vector=[0.1, 0.2],
            top_k=1,
            filters={"tenant_id": "tenant123"},
        )


class TestBM25Retriever:
    """Tests for BM25Retriever."""

    def test_bm25_keyword_retrieval(self):
        c1 = Chunk(
            chunk_id="c1",
            doc_id="d1",
            chunk_index=0,
            text="Asthma inhaler",
            token_count=2,
            tenant_id="tenant123",
            provider_id="provider456",
        )
        c2 = Chunk(
            chunk_id="c2",
            doc_id="d2",
            chunk_index=0,
            text="Diabetes insulin",
            token_count=2,
            tenant_id="tenant123",
            provider_id="provider456",
        )
        chunks = [c1, c2]

        retriever = BM25Retriever(chunks=chunks)
        results = retriever.retrieve("asthma inhalers", top_k=1, filters={"provider_id": "provider456"})

        assert len(results) == 1
        assert results[0].chunk_id == "c1"


class TestEnsembleRetriever:
    """Tests for EnsembleRetriever (RRF)."""

    def test_reciprocal_rank_fusion(self):
        c1 = Chunk(
            chunk_id="c1",
            doc_id="d1",
            chunk_index=0,
            text="Text 1",
            token_count=2,
            tenant_id="tenant123",
            provider_id="provider456",
        )
        c2 = Chunk(
            chunk_id="c2",
            doc_id="d1",
            chunk_index=1,
            text="Text 2",
            token_count=2,
            tenant_id="tenant123",
            provider_id="provider456",
        )

        mock_ret1 = MagicMock()
        mock_ret1.retrieve.return_value = [c1, c2]

        mock_ret2 = MagicMock()
        mock_ret2.retrieve.return_value = [c2, c1]

        ensemble = EnsembleRetriever(retrievers=[mock_ret1, mock_ret2], rrf_k=60)
        results = ensemble.retrieve("query", top_k=2, filters={"tenant_id": "tenant123"})

        assert len(results) == 2
        # RRF scores should tie, returning unique chunks
        retrieved_ids = {r.chunk_id for r in results}
        assert retrieved_ids == {"c1", "c2"}
        mock_ret1.retrieve.assert_called_once_with("query", top_k=4, filters={"tenant_id": "tenant123"})


class TestLocalReranker:
    """Tests for LocalReranker."""

    @patch("mentera_rag.retrieval.rerankers.HAS_CROSS_ENCODER", True)
    @patch("mentera_rag.retrieval.rerankers.CrossEncoder")
    def test_local_reranker_cuda_device_selection(self, mock_cross_encoder_class):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.2, 0.9]
        mock_cross_encoder_class.return_value = mock_model

        c1 = Chunk(
            chunk_id="c1",
            doc_id="d1",
            chunk_index=0,
            text="Irrelevant context",
            token_count=2,
            tenant_id="tenant123",
            provider_id="provider456",
        )
        c2 = Chunk(
            chunk_id="c2",
            doc_id="d1",
            chunk_index=1,
            text="Relevant context",
            token_count=2,
            tenant_id="tenant123",
            provider_id="provider456",
        )
        chunks = [c1, c2]

        reranker = LocalReranker(model_name="BAAI/bge-reranker-v2-m3", device="cpu")
        reranked = reranker.rerank("medical query", chunks, top_n=1)

        assert len(reranked) == 1
        assert reranked[0].chunk_id == "c2"


class TestBedrockReranker:
    """Tests for BedrockReranker."""

    @patch("boto3.client")
    def test_bedrock_rerank_invocation(self, mock_boto_client):
        mock_bedrock = MagicMock()
        resp_data = {"results": [{"index": 1, "relevance_score": 0.95}]}
        mock_response = {"body": MagicMock(read=lambda: json.dumps(resp_data).encode())}
        mock_bedrock.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_bedrock

        c1 = Chunk(
            chunk_id="c1",
            doc_id="d1",
            chunk_index=0,
            text="First text",
            token_count=2,
            tenant_id="tenant123",
            provider_id="provider456",
        )
        c2 = Chunk(
            chunk_id="c2",
            doc_id="d1",
            chunk_index=1,
            text="Second text",
            token_count=2,
            tenant_id="tenant123",
            provider_id="provider456",
        )
        chunks = [c1, c2]

        reranker = BedrockReranker(model_name="amazon.rerank-v1:0")
        results = reranker.rerank("search query", chunks, top_n=1)

        assert len(results) == 1
        assert results[0].chunk_id == "c2"
        mock_bedrock.invoke_model.assert_called_once()
