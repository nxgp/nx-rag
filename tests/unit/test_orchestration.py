from unittest.mock import MagicMock

import pytest

from mentera_rag.chunking.schemas import Chunk
from mentera_rag.orchestration.agentic import AgenticRAGGraph
from mentera_rag.orchestration.linear import LinearRAGPipeline


@pytest.fixture
def mock_chunks():
    c1 = Chunk(
        chunk_id="doc1_c0",
        doc_id="doc1",
        chunk_index=0,
        text="Target matched chunk number one.",
        tenant_id="tenant123",
        provider_id="provider456",
    )
    c2 = Chunk(
        chunk_id="doc1_c1",
        doc_id="doc1",
        chunk_index=1,
        text="Secondary context chunk text here.",
        tenant_id="tenant123",
        provider_id="provider456",
    )
    return [c1, c2]


@pytest.mark.unit
def test_linear_rag_pipeline_run(mock_chunks):
    """Test LinearRAGPipeline runs retrieval successfully and omits generation."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = mock_chunks

    pipeline = LinearRAGPipeline(retriever=mock_retriever)
    result = pipeline.run(
        query="test query",
        top_k=2,
        top_n=2,
        filters={"tenant_id": "tenant123", "provider_id": "provider456"},
    )

    assert result["pipeline_type"] == "linear"
    assert result["query"] == "test query"
    assert result["answer"] == ""  # Context-only pipeline returns empty answer
    assert len(result["retrieved_chunks"]) == 2
    assert result["retrieved_chunks"][0].chunk_id == "doc1_c0"

    mock_retriever.retrieve.assert_called_once_with(
        "test query",
        top_k=2,
        filters={"tenant_id": "tenant123", "provider_id": "provider456"},
    )


@pytest.mark.unit
def test_agentic_rag_graph_run(mock_chunks):
    """Test AgenticRAGGraph runs query rewriting and returns context without generation."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = mock_chunks

    # Mock LLM for query rewriting
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "rewritten query"

    pipeline = AgenticRAGGraph(
        retriever=mock_retriever,
        llm_provider=mock_llm,
        max_retries=1,
    )

    # 1. Test when relevance check matches (no rewrite)
    # The grade documents node has overlap: "test query" and matched chunks have "chunk"
    result = pipeline.run(
        query="chunk text",
        filters={"tenant_id": "tenant123", "provider_id": "provider456"},
    )

    assert result["pipeline_type"] == "agentic"
    assert result["answer"] == ""
    assert len(result["retrieved_chunks"]) == 2
    assert result["retry_count"] == 0

    # 2. Test when relevance fails and triggers query rewrite
    # "unmatched" query has no overlap with chunks
    result_rewrite = pipeline.run(
        query="unmatched",
        filters={"tenant_id": "tenant123"},
    )
    assert result_rewrite["retry_count"] == 1
    assert result_rewrite["final_query"] == "rewritten query"
    assert result_rewrite["answer"] == ""
