"""
End-to-End Integration Test Suite for Mentera RAG Pipeline.

Verifies end-to-end integration across Ingestion, Chunking, Embeddings, Vector Store Indexing,
Retrieval, and Orchestration, with strict verification of tenant isolation.
"""

from unittest.mock import MagicMock, patch

from mentera_rag.chunking.recursive import RecursiveCharacterChunker
from mentera_rag.chunking.schemas import Chunk
from mentera_rag.chunking.schemas import Document as ChunkDocument
from mentera_rag.evaluation.evaluator import RAGEvaluator
from mentera_rag.orchestration.agentic import AgenticRAGGraph
from mentera_rag.orchestration.linear import LinearRAGPipeline
from mentera_rag.retrieval.bm25 import BM25Retriever
from mentera_rag.retrieval.dense import DenseRetriever
from mentera_rag.retrieval.ensemble import EnsembleRetriever
from mentera_rag.vector_stores.factory import VectorStoreFactory


class TestE2EPipelineIntegration:
    """Integration test suite for the complete pipeline execution flow."""

    @patch("mentera_rag.vector_stores.qdrant_store.QdrantClient")
    def test_e2e_linear_pipeline_flow(self, mock_qdrant_class):
        # 1. Setup Vector Store Mock
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.collection_exists.return_value = True

        mock_point = MagicMock()
        mock_point.id = "c1"
        mock_point.score = 0.92
        mock_point.payload = {
            "chunk_id": "c1",
            "doc_id": "doc1",
            "text": "Mitochondria undergo structural changes.",
            "tenant_id": "tenant_a",
            "provider_id": "provider_x",
        }
        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_qdrant_client.query_points.return_value = mock_response
        mock_qdrant_class.return_value = mock_qdrant_client

        # 2. Ingestion & Chunking with tenant data
        chunker = RecursiveCharacterChunker(chunk_size=300, chunk_overlap=30)
        doc = ChunkDocument(
            doc_id="doc1",
            content="Mitochondria undergo structural changes.",
            source="pubmed",
            tenant_id="tenant_a",
            provider_id="provider_x",
            metadata={},
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0
        assert chunks[0].tenant_id == "tenant_a"

        # 3. Setup Mock Embeddings
        mock_embed = MagicMock()
        mock_embed.dimension = 256
        mock_embed.embed_query.return_value = [0.1] * 256

        # 4. Resolve Vector Store & Retrievers
        vstore = VectorStoreFactory.get_vector_store("e2e_test_col", mock_embed.dimension)
        dense_ret = DenseRetriever(embed_provider=mock_embed, vector_store=vstore)
        bm25_ret = BM25Retriever(chunks=chunks)
        retriever = EnsembleRetriever(retrievers=[dense_ret, bm25_ret], rrf_k=60)

        # 5. Orchestration: Linear Pipeline
        pipeline = LinearRAGPipeline(retriever=retriever)
        res = pipeline.run(
            "What role do mitochondria play?",
            filters={"tenant_id": "tenant_a", "provider_id": "provider_x"},
        )

        assert res["pipeline_type"] == "linear"
        assert res["answer"] == ""  # Context-only returns empty answer
        assert len(res["retrieved_chunks"]) > 0
        assert res["retrieved_chunks"][0].tenant_id == "tenant_a"

        # 6. Evaluation Framework Mock Logging
        mock_tracker = MagicMock()
        mock_tracker.log_evaluation_run.return_value = "e2e-run-linear-001"
        evaluator = RAGEvaluator(tracker=mock_tracker)
        test_cases = [{"query": "What role do mitochondria play?", "doc_id": "doc1"}]

        eval_res = evaluator.evaluate_pipeline(
            pipeline=pipeline,
            test_cases=test_cases,
            run_name="e2e_linear_test",
            pipeline_params={"pipeline_type": "linear"},
        )
        assert eval_res["run_id"] == "e2e-run-linear-001"

    @patch("mentera_rag.vector_stores.qdrant_store.QdrantClient")
    def test_e2e_agentic_pipeline_flow(self, mock_qdrant_class):
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.collection_exists.return_value = True
        mock_point = MagicMock()
        mock_point.id = "c1"
        mock_point.score = 0.95
        mock_point.payload = {
            "chunk_id": "c1",
            "doc_id": "doc1",
            "text": "Mitochondrial permeability transition.",
            "tenant_id": "tenant_a",
            "provider_id": "provider_x",
        }
        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_qdrant_client.query_points.return_value = mock_response
        mock_qdrant_class.return_value = mock_qdrant_client

        # Chunks for BM25
        chunk = Chunk(
            chunk_id="c1",
            doc_id="doc1",
            chunk_index=0,
            text="Mitochondrial permeability transition.",
            tenant_id="tenant_a",
            provider_id="provider_x",
        )
        chunks = [chunk]

        mock_embed = MagicMock()
        mock_embed.dimension = 256
        mock_embed.embed_query.return_value = [0.1] * 256

        vstore = VectorStoreFactory.get_vector_store("e2e_agentic_col", mock_embed.dimension)
        dense_ret = DenseRetriever(embed_provider=mock_embed, vector_store=vstore)
        bm25_ret = BM25Retriever(chunks=chunks)
        retriever = EnsembleRetriever(retrievers=[dense_ret, bm25_ret], rrf_k=60)

        # Mock LLM for query rewriter
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "rewritten query"

        agentic_graph = AgenticRAGGraph(retriever=retriever, llm_provider=mock_llm, max_retries=1)
        res = agentic_graph.run(
            "How do mitochondria initiate cell death?",
            filters={"tenant_id": "tenant_a", "provider_id": "provider_x"},
        )

        assert res["pipeline_type"] == "agentic"
        assert res["answer"] == ""
        assert len(res["retrieved_chunks"]) > 0
        assert res["retrieved_chunks"][0].tenant_id == "tenant_a"

    def test_tenant_isolation_leakage_prevention(self):
        """
        Verify that queries for tenant_a return ONLY tenant_a chunks,
        and do not leak any records belonging to tenant_b.
        """
        # Create chunks from separate tenants
        c_a = Chunk(
            chunk_id="c_a",
            doc_id="doc_a",
            chunk_index=0,
            text="Sensitive patient context for tenant A",
            tenant_id="tenant_a",
            provider_id="provider_x",
        )
        c_b = Chunk(
            chunk_id="c_b",
            doc_id="doc_b",
            chunk_index=0,
            text="Private records for tenant B",
            tenant_id="tenant_b",
            provider_id="provider_y",
        )
        chunks = [c_a, c_b]

        # Initialize BM25Retriever containing both chunks
        retriever = BM25Retriever(chunks=chunks)

        # 1. Query for tenant_a
        results_a = retriever.retrieve(
            "patient records",
            top_k=10,
            filters={"tenant_id": "tenant_a", "provider_id": "provider_x"},
        )
        assert len(results_a) == 1
        assert results_a[0].chunk_id == "c_a"
        assert results_a[0].text == "Sensitive patient context for tenant A"

        # 2. Query for tenant_b
        results_b = retriever.retrieve(
            "patient records",
            top_k=10,
            filters={"tenant_id": "tenant_b", "provider_id": "provider_y"},
        )
        assert len(results_b) == 1
        assert results_b[0].chunk_id == "c_b"
        assert results_b[0].text == "Private records for tenant B"

        # 3. Query with non-existent tenant filters
        results_none = retriever.retrieve(
            "patient records",
            top_k=10,
            filters={"tenant_id": "tenant_nonexistent", "provider_id": "provider_x"},
        )
        assert len(results_none) == 0
