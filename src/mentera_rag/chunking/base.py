"""
Abstract Base Class for all Chunking Strategies — Mentera RAG Pipeline.

Concrete chunkers inherit from BaseChunker and implement `chunk()`.
The `_build_chunk` helper ensures consistent chunk_id formatting and
full metadata propagation from Document → Chunk (including tenant fields).
"""

from abc import ABC, abstractmethod
from typing import Any

from mentera_rag.chunking.schemas import Chunk, Document


class BaseChunker(ABC):
    """
    Abstract Base Class for all chunking strategies.

    Every chunker (Recursive, Semantic, Fixed, etc.) must implement `chunk()`.
    Use `_build_chunk()` to construct Chunk objects — it guarantees all tenant
    and provenance fields are correctly propagated from the parent Document.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Splits a Document into a list of Chunk objects.

        Args:
            document: Input Document with full tenant metadata.

        Returns:
            Ordered list of Chunk objects ready for embedding and indexing.
        """
        pass

    def _build_chunk(
        self,
        document: Document,
        chunk_text: str,
        chunk_index: int,
        start_char: int | None = None,
        end_char: int | None = None,
        page_number: int | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        """
        Construct a Chunk with consistent ID and full metadata inheritance.

        Propagates all tenant isolation fields (tenant_id, provider_id, patient_id),
        document classification fields (document_type, collection_name, tags,
        upload_timestamp, file_hash), and positional metadata from the parent Document.

        Args:
            document: Parent Document providing identity and tenant context.
            chunk_text: The text content for this chunk.
            chunk_index: Zero-based index of this chunk within the document.
            start_char: Character offset where this chunk starts in document.content.
            end_char: Character offset where this chunk ends in document.content.
            page_number: Source page number (from PDF/multi-page parsers).
            extra_metadata: Strategy-specific metadata (e.g. chunk_size_target, strategy).

        Returns:
            A fully populated Chunk object.
        """
        chunk_id = f"{document.doc_id}_c{chunk_index}"

        # Merge source metadata + extra metadata from chunker strategy
        chunk_metadata: dict[str, Any] = {
            "source": document.source,
            **document.metadata,
            **(extra_metadata or {}),
        }

        return Chunk(
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            text=chunk_text,
            chunk_index=chunk_index,
            # Tenant isolation — propagated from Document
            tenant_id=document.tenant_id,
            provider_id=document.provider_id,
            patient_id=document.patient_id,
            # Document classification — propagated from Document
            document_type=document.document_type,
            collection_name=document.collection_name,
            tags=document.tags,
            upload_timestamp=document.upload_timestamp,
            file_hash=document.file_hash,
            # Positional metadata
            start_char=start_char,
            end_char=end_char,
            page_number=page_number,
            token_count=len(chunk_text.split()),
            # embedding_model filled in later by EmbeddingFactory
            embedding_model=None,
            metadata=chunk_metadata,
        )
