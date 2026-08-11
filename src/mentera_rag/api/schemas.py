"""
FastAPI Request and Response Schemas — Mentera RAG Pipeline.

Establishes strict validation boundaries for `/upload/presign`, `/ingest`,
and `/query` API endpoints.
"""

from typing import Any
from pydantic import BaseModel, Field


class PresignRequest(BaseModel):
    filename: str = Field(..., description="Original name of the file to be uploaded")
    content_type: str = Field(
        default="application/octet-stream",
        description="MIME type of the file (e.g. application/pdf, text/plain)",
    )
    tenant_id: str = Field(..., description="Organization / tenant identifier")
    provider_id: str = Field(..., description="Data provider identifier")


class PresignResponse(BaseModel):
    upload_url: str = Field(..., description="Presigned upload URL (PUT method)")
    storage_key: str = Field(..., description="Unique key/path to pass to the /ingest endpoint")
    expires_in: int = Field(default=3600, description="Expiration time in seconds")


class IngestRequest(BaseModel):
    storage_key: str = Field(
        ...,
        description="Unique path key returned from /upload/presign after upload is complete",
    )
    tenant_id: str = Field(..., description="Organization / tenant identifier")
    provider_id: str = Field(..., description="Data provider identifier")
    patient_id: str | None = Field(default=None, description="Optional patient scoping filter")
    tags: list[str] = Field(default_factory=list, description="Metadata tags for filtering")
    collection_name: str = Field(default="default", description="Namespace within tenant")


class IngestResponse(BaseModel):
    document_id: str = Field(..., description="Deteminisitic document ID assigned during ingestion")
    chunk_count: int = Field(..., description="Total text chunks indexed in Qdrant")
    embedding_model: str = Field(..., description="Embedding model used for indexing")
    status: str = Field(default="success", description="Ingestion processing status")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Semantic search query string")
    tenant_id: str = Field(..., description="Organization / tenant identifier (required)")
    provider_id: str = Field(..., description="Data provider identifier (required)")
    patient_id: str | None = Field(default=None, description="Optional patient scoping filter")
    collection_name: str | None = Field(default=None, description="Optional collection filter")
    tags: list[str] | None = Field(default=None, description="Optional metadata tag filter list")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to retrieve")


class RetrievedContext(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Parent document identifier")
    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Relevance score (higher is more relevant)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata payload")


class QueryResponse(BaseModel):
    status: str = Field(default="success", description="Status code")
    query: str = Field(..., description="Original search query")
    retrieved_contexts: list[RetrievedContext] = Field(
        ..., description="Standardized list of relevant document chunks"
    )
    total_results: int = Field(..., description="Number of retrieved chunks")
    latency_ms: float = Field(..., description="Query execution latency in milliseconds")
