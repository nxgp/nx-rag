"""
Chunking Data Contracts — Mentera RAG Pipeline.

Document is the input to chunkers (produced by UploadPipeline from parsed text).
Chunk is the unit stored in Qdrant — every field is a searchable/filterable payload.

Tenant isolation fields (tenant_id, provider_id, patient_id) are first-class fields
on both Document and Chunk so they propagate reliably into Qdrant payload.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Input document to the chunking layer.

    Produced by UploadPipeline after parsing raw files. Carries all tenant
    isolation metadata that must be propagated down into every Chunk.
    """

    doc_id: str = Field(
        ...,
        description="Unique document identifier (e.g. SHA-256 prefix of content)",
    )
    content: str = Field(..., description="Full parsed text content of the document")
    source: str = Field(
        ...,
        description="Origin identifier (filename, storage key, or URL)",
    )

    # Multi-tenancy (mandatory)
    tenant_id: str = Field(..., description="Organization / tenant identifier")
    provider_id: str = Field(..., description="Data provider identifier within the tenant")
    patient_id: str | None = Field(default=None, description="Optional patient-level scoping")

    # Document classification
    document_type: str = Field(
        default="txt",
        description="Parsed file type: 'pdf', 'txt', 'md', 'image'",
    )
    collection_name: str = Field(
        default="default",
        description="Logical namespace / grouping within the tenant",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="User-provided labels for filtering",
    )
    upload_timestamp: datetime | None = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of ingestion",
    )
    file_hash: str = Field(
        default="",
        description="SHA-256 content hash of the original file",
    )

    # Optional document-level metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional metadata (page_count, language, etc.)",
    )


class Chunk(BaseModel):
    """
    A text segment extracted from a parent Document.

    This is the atomic unit embedded and stored in Qdrant. Every field that
    should be filterable is a top-level payload key in Qdrant (not buried in metadata).
    """

    # Core identity
    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier: '{doc_id}_c{chunk_index}'",
    )
    doc_id: str = Field(..., description="Foreign key to parent Document.doc_id")
    text: str = Field(..., description="Text content of this chunk")
    score: float = Field(default=0.0, description="Relevance / similarity score from retrieval")
    chunk_index: int = Field(
        ..., description="Zero-based sequential index within the parent document"
    )

    # Multi-tenancy (propagated from parent Document, indexed in Qdrant)
    tenant_id: str = Field(..., description="Organization / tenant identifier")
    provider_id: str = Field(..., description="Data provider identifier")
    patient_id: str | None = Field(default=None, description="Optional patient-level scoping")

    # Document context (propagated, filterable)
    document_type: str = Field(
        default="txt",
        description="Source file type: 'pdf', 'txt', 'md', 'image'",
    )
    collection_name: str = Field(
        default="default",
        description="Logical namespace within the tenant",
    )
    tags: list[str] = Field(default_factory=list, description="User-provided labels")
    upload_timestamp: datetime | None = Field(
        default_factory=datetime.utcnow, description="UTC ingestion timestamp"
    )
    file_hash: str = Field(default="", description="SHA-256 of the source file")

    # Positional metadata
    start_char: int | None = Field(
        default=None, description="Character offset where chunk starts in original content"
    )
    end_char: int | None = Field(
        default=None, description="Character offset where chunk ends in original content"
    )
    page_number: int | None = Field(
        default=None, description="Source page number (PDFs and multi-page documents)"
    )
    token_count: int | None = Field(
        default=None, description="Approximate word-token count for context-window checks"
    )

    # Chunking provenance
    parent_chunk_id: str | None = Field(
        default=None,
        description="Parent chunk ID for hierarchical / parent-child chunking strategies",
    )

    # Embedding provenance (set by EmbeddingFactory after embedding)
    embedding_model: str | None = Field(
        default=None,
        description="Embedding model ID used to produce the vector (for auditability)",
    )

    # Catch-all for extra metadata from chunker strategies
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional payload (strategy, chunk_size_target, source, etc.)",
    )
