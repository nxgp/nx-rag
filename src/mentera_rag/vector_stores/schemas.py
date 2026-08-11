"""
Schemas for Vector Store Operations — Mentera RAG Pipeline.

VectorSearchResult is the unified result type returned by QdrantVectorStore.
It includes all filterable tenant and document classification fields as
first-class attributes so callers don't need to dig into the metadata dict.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class VectorSearchResult(BaseModel):
    """
    Standardized search result returned by the Qdrant vector store.

    All tenant isolation fields are first-class attributes (not buried in metadata)
    so the retrieval layer can apply post-retrieval tenant verification if needed.
    """

    # Core content
    chunk_id: str = Field(..., description="Unique identifier of the matching chunk")
    doc_id: str = Field(..., description="ID of the parent document")
    text: str = Field(..., description="Raw text content of the chunk")
    score: float = Field(..., description="Similarity or relevance score (higher = more relevant)")

    # Tenant isolation (indexed Qdrant payload fields)
    tenant_id: str = Field(default="", description="Organization / tenant identifier")
    provider_id: str = Field(default="", description="Data provider identifier")
    patient_id: str | None = Field(default=None, description="Optional patient scoping")

    # Document classification (filterable payload fields)
    document_type: str = Field(default="", description="Source file type: pdf, txt, md, image")
    collection_name: str = Field(default="default", description="Logical namespace within tenant")
    tags: list[str] = Field(default_factory=list, description="User-provided labels")
    upload_timestamp: datetime | None = Field(default=None, description="UTC ingestion timestamp")
    page_number: int | None = Field(default=None, description="Source page number in document")

    # Extensible metadata (strategy, source, char offsets, etc.)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional payload metadata from chunker (strategy, start_char, etc.)",
    )
