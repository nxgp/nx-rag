"""
Upload & Ingestion Pipeline — Mentera RAG Pipeline.

Orchestrates the entire flow of document ingestion:
  1. Download raw file bytes from Object Storage
  2. Compute SHA-256 hash for deduplication
  3. Query Qdrant to check if the file was already indexed for this tenant
  4. Parse the file into pages using ParserFactory (PDF, TXT, MD, Images)
  5. Chunk pages recursively using RecursiveCharacterChunker
  6. Generate embeddings using EmbeddingFactory (AWS, Azure, GCP)
  7. Index the chunks and vectors into Qdrant with full multi-tenant payloads
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any

from mentera_rag.chunking.recursive import RecursiveCharacterChunker
from mentera_rag.chunking.schemas import Chunk, Document
from mentera_rag.config.settings import settings
from mentera_rag.embeddings.factory import EmbeddingFactory
from mentera_rag.parsing.factory import SUPPORTED_EXTENSIONS, ParserFactory
from mentera_rag.storage.factory import StorageFactory
from mentera_rag.vector_stores.factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class UploadPipeline:
    """
    Ingestion pipeline that downloads a file from storage, parses, chunks,
    embeds, and indexes it into Qdrant.
    """

    def __init__(
        self,
        storage_provider: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
    ):
        """Initialize pipeline adapters."""
        self.storage = StorageFactory.get_store(storage_provider)
        self.embedding = EmbeddingFactory.get_provider(
            provider_type=embedding_provider,
            model_name=embedding_model,
            dimension=embedding_dimension,
        )
        self.vector_store = VectorStoreFactory.get_vector_store()
        self.vector_store.create_collection()  # Ensure Qdrant collection and payload indexes exist

    def run(
        self,
        storage_key: str,
        tenant_id: str,
        provider_id: str,
        patient_id: str | None = None,
        tags: list[str] | None = None,
        collection_name: str = "default",
    ) -> dict[str, Any]:
        """
        Run the ingestion pipeline on a file key from the object store.

        Args:
            storage_key: Storage key/path of the file in the bucket.
            tenant_id: Tenant owner.
            provider_id: Data provider ID.
            patient_id: Optional patient filter key.
            tags: Metadata tags.
            collection_name: Qdrant logical namespace.

        Returns:
            Dict containing ingestion summary (doc_id, chunk_count, duration, status).
        """
        start_time = time.perf_counter()
        tags_list = tags or []

        # 1. Resolve file extension & validate
        filename = os.path.basename(storage_key)
        _, ext = os.path.splitext(filename)
        ext = ext.lower().strip()

        if not ParserFactory.is_supported(ext):
            raise ValueError(
                f"Unsupported file extension '{ext}' for ingestion. "
                f"Supported types: {list(SUPPORTED_EXTENSIONS)}"
            )

        # 2. Download file from storage
        logger.info("Downloading file from storage: %s", storage_key)
        file_bytes = self.storage.download(storage_key)

        if not file_bytes:
            raise FileNotFoundError(
                f"File '{storage_key}' in storage contains 0 bytes. "
                "Ensure the file was uploaded to the presigned upload URL before ingestion."
            )

        # 3. Compute SHA-256 file hash & document ID
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        doc_id = f"doc_{tenant_id}_{file_hash[:16]}"

        # 4. Deduplication Check
        # Check if this file_hash has already been indexed for this tenant in Qdrant
        try:
            from qdrant_client import models

            client = getattr(self.vector_store, "client", None)
            if client is not None:
                cond1 = models.FieldCondition(
                    key="tenant_id", match=models.MatchValue(value=tenant_id)
                )
                cond2 = models.FieldCondition(
                    key="file_hash", match=models.MatchValue(value=file_hash)
                )
                filter_conditions: list[Any] = [cond1, cond2]
                scroll_result = client.scroll(
                    collection_name=self.vector_store.collection_name,
                    scroll_filter=models.Filter(must=filter_conditions),
                    limit=1,
                    with_payload=True,
                )
                if scroll_result and scroll_result[0]:
                    existing_record = scroll_result[0][0]
                    existing_payload = existing_record.payload or {}
                    logger.info(
                        "Document '%s' (hash: %s) already indexed for tenant '%s'.",
                        filename,
                        file_hash,
                        tenant_id,
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    return {
                        "document_id": existing_payload.get("doc_id", doc_id),
                        "chunk_count": 0,  # 0 new chunks indexed
                        "embedding_model": existing_payload.get(
                            "embedding_model", self.embedding.model_name
                        ),
                        "status": "already_indexed",
                        "processing_time_ms": round(elapsed_ms, 2),
                    }
        except Exception as e:
            logger.warning("Deduplication check failed, proceeding with ingestion: %s", e)

        # 5. Write to local temp directory for parsing
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        temp_filepath = upload_dir / f"{doc_id}{ext}"

        try:
            temp_filepath.write_bytes(file_bytes)

            # 6. Parse document
            parser = ParserFactory.get_parser(ext)
            logger.info("Parsing file '%s' using %s", filename, parser.__class__.__name__)
            parsed_pages = parser.parse(temp_filepath)

            if not parsed_pages:
                raise ValueError(f"Document parsing returned no extractable text for: {filename}")

            # 7. Chunk parsed pages
            chunker = RecursiveCharacterChunker(
                chunk_size=settings.DEFAULT_CHUNK_SIZE,
                chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
            )

            all_chunks: list[Chunk] = []
            chunk_global_index = 0

            for page in parsed_pages:
                # Wrap each parsed page into a Document schema
                doc = Document(
                    doc_id=doc_id,
                    content=page.content,
                    source=filename,
                    tenant_id=tenant_id,
                    provider_id=provider_id,
                    patient_id=patient_id,
                    document_type=ext.lstrip("."),
                    collection_name=collection_name,
                    tags=tags_list,
                    file_hash=file_hash,
                    metadata=page.metadata,
                )

                page_chunks = chunker.chunk(doc)

                # Assign page numbers and unique IDs, set embedding model type
                for chunk in page_chunks:
                    chunk.page_number = page.page_number
                    chunk.embedding_model = self.embedding.model_name
                    # Make chunk_id globally unique across pages
                    chunk.chunk_id = f"{doc_id}_c{chunk_global_index}"
                    chunk.chunk_index = chunk_global_index
                    chunk_global_index += 1

                all_chunks.extend(page_chunks)

            logger.info("Generated %d chunks from document: %s", len(all_chunks), filename)

            # 8. Embed chunks
            chunk_texts = [chunk.text for chunk in all_chunks]
            logger.info("Generating embeddings using model: %s", self.embedding.model_name)
            vectors = self.embedding.embed_documents(chunk_texts)

            # 9. Index in Qdrant
            logger.info(
                "Indexing chunks into Qdrant collection: %s", self.vector_store.collection_name
            )
            self.vector_store.index_chunks(all_chunks, vectors)

        finally:
            # 10. Clean up temp staging file
            if temp_filepath.exists():
                temp_filepath.unlink()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "document_id": doc_id,
            "chunk_count": len(all_chunks),
            "embedding_model": self.embedding.model_name,
            "status": "success",
            "processing_time_ms": round(elapsed_ms, 2),
        }
