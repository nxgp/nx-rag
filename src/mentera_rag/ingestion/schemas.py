"""
Canonical Data Contracts for Ingestion — Mentera RAG Pipeline.

These Pydantic models establish the strict schema boundary between raw document
uploads and downstream RAG modules (Parsing, Chunking, Vector Stores, Evaluation).

Multi-tenancy is enforced via mandatory tenant_id and provider_id fields on
every Document. These propagate through chunking into Qdrant payload for
per-tenant filtered retrieval.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """
    Represents a raw parsed document ready for chunking and embedding.

    Produced by the UploadPipeline after file parsing. Contains mandatory
    tenant isolation fields that propagate through to Qdrant payload.

    TODO: Implement PII/PHI scrubbing (de-identification) at this layer
    BEFORE chunking and sending text to third-party embedding APIs (HIPAA/GDPR).
    """

    model_config = ConfigDict(frozen=True)

    # Core identity
    id: str = Field(..., description="Deterministic unique document ID (e.g. SHA-256 prefix + filename)")
    title: str = Field(default="", description="Document title or filename")
    text: str = Field(..., description="Full parsed text content of the document")
    source: str = Field(..., description="Original filename or storage key")

    # Multi-tenancy (mandatory for all uploaded documents)
    tenant_id: str = Field(..., description="Organization / tenant identifier (required)")
    provider_id: str = Field(..., description="Data provider identifier within the tenant (required)")
    patient_id: str | None = Field(default=None, description="Optional patient-level scoping")

    # Document classification & storage
    document_type: str = Field(
        ...,
        description="Parsed file type: 'pdf', 'txt', 'md', 'image'",
    )
    storage_key: str = Field(
        default="",
        description="Cloud storage object key where the raw file is stored",
    )
    file_hash: str = Field(
        default="",
        description="SHA-256 content hash of the original file (used for deduplication)",
    )

    # Timing & tagging
    upload_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the document was ingested",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="User-provided labels for filtering (e.g. ['cardiology', '2024'])",
    )
    collection_name: str = Field(
        default="default",
        description="Logical namespace within the tenant for grouping documents",
    )

    # Extensible metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value metadata (page_count, language, etc.)",
    )


class DatasetManifest(BaseModel):
    """
    Manifest documenting the version, counts, and checksums of a batch ingestion run.
    """

    dataset_name: str = Field(..., description="Name of the ingested dataset or upload batch")
    version: str = Field(..., description="Pipeline / dataset version tag")
    document_count: int = Field(..., description="Total documents ingested")
    sha256_checksums: dict[str, str] = Field(..., description="Map of filename to SHA-256 hash")
