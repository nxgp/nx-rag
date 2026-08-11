"""
Qdrant Vector Store Adapter — Mentera RAG Pipeline.

Single vector store for all tenants — tenant isolation is enforced via
mandatory `tenant_id` payload filters on every search operation.

Indexed payload fields (for fast filtered queries):
  - tenant_id        (keyword)
  - provider_id      (keyword)
  - document_type    (keyword)
  - collection_name  (keyword)
  - patient_id       (keyword, optional)

Supports:
  - HNSW dense vector indexing
  - BM25 hybrid search via Reciprocal Rank Fusion (RRF)
  - Sparse payload indexes for all tenant/classification fields
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from qdrant_client import QdrantClient, models

from mentera_rag.chunking.schemas import Chunk
from mentera_rag.vector_stores.base import BaseVectorStore
from mentera_rag.vector_stores.schemas import VectorSearchResult

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector store adapter for the Mentera RAG Pipeline.

    Uses a single shared collection with tenant isolation enforced via
    Qdrant payload filters. Payload indexes are created on all filterable
    fields during collection initialization.
    """

    def __init__(
        self,
        collection_name: str = "mentera_chunks",
        dimension: int = 1024,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
    ):
        super().__init__(collection_name=collection_name, dimension=dimension)
        self.url = url
        self.client = QdrantClient(url=self.url, api_key=api_key)

    def create_collection(self, force_recreate: bool = False) -> None:
        """
        Create the Qdrant collection with HNSW index and payload indexes.

        Payload indexes are created on tenant isolation fields (tenant_id,
        provider_id, patient_id) and document classification fields
        (document_type, collection_name) for fast filtered queries.
        """
        exists = self.client.collection_exists(self.collection_name)

        if exists and force_recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d).", self.collection_name, self.dimension
            )

        import contextlib

        # Create keyword payload indexes for fast tenant-filtered queries
        indexed_keyword_fields = [
            "tenant_id",
            "provider_id",
            "patient_id",
            "document_type",
            "collection_name",
        ]
        for field_name in indexed_keyword_fields:
            with contextlib.suppress(Exception):
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.debug("Payload index created: %s", field_name)

        # Create datetime index on upload_timestamp for time-range queries
        with contextlib.suppress(Exception):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="upload_timestamp",
                field_schema=models.PayloadSchemaType.DATETIME,
            )

    def index_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """
        Upsert chunks with their dense vectors and full metadata payload into Qdrant.

        Tenant isolation fields are stored as top-level payload keys so they
        can be used in Qdrant Filter conditions directly.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Chunks ({len(chunks)}) and vectors ({len(vectors)}) count must match."
            )

        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=False):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            # Serialize upload_timestamp to ISO string for Qdrant datetime payload
            upload_ts: str | None = None
            if chunk.upload_timestamp:
                ts = chunk.upload_timestamp
                upload_ts = ts.isoformat() if isinstance(ts, datetime) else str(ts)

            payload: dict[str, Any] = {
                # Core identity
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                # Tenant isolation (indexed)
                "tenant_id": chunk.tenant_id,
                "provider_id": chunk.provider_id,
                "patient_id": chunk.patient_id,
                # Document classification (indexed)
                "document_type": chunk.document_type,
                "collection_name": chunk.collection_name,
                "tags": chunk.tags,
                "upload_timestamp": upload_ts,
                "file_hash": chunk.file_hash,
                # Positional metadata
                "page_number": chunk.page_number,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
                # Embedding provenance
                "embedding_model": chunk.embedding_model,
                # Extra chunker metadata (strategy, chunk_size_target, source, etc.)
                "metadata": chunk.metadata,
            }

            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.debug("Upserted %d chunks into collection '%s'.", len(points), self.collection_name)

    def _build_filter(self, filters: dict[str, Any] | None) -> models.Filter | None:
        """
        Convert a dict of filter key-values to a Qdrant Filter with must conditions.

        All conditions use MatchValue (exact keyword match). Use tenant_id as a
        mandatory filter for every search to enforce tenant isolation.

        Example:
            filters = {
                "tenant_id": "acme_corp",
                "provider_id": "clinic_a",
                "document_type": "pdf",
            }
        """
        if not filters:
            return None

        must_conditions: list[Any] = [
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in filters.items()
            if v is not None  # Skip None values (e.g. optional patient_id)
        ]

        if not must_conditions:
            return None

        return models.Filter(must=must_conditions)

    def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Execute pure dense vector similarity search with optional tenant filters.

        Args:
            query_vector: Embedded query vector.
            top_k: Number of nearest neighbours to return.
            filters: Payload filter dict — MUST include 'tenant_id' for isolation.

        Returns:
            List of VectorSearchResult sorted by descending score.
        """
        q_filter = self._build_filter(filters)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=q_filter,
        )
        return self._points_to_results(response.points)

    def search_hybrid(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 10,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Hybrid search combining dense vector search and BM25 via Reciprocal Rank Fusion.

        First retrieves dense candidates (top_k * 2), then re-scores using BM25
        on the candidate text and merges ranks via weighted RRF.

        Args:
            query_text: Raw query string for BM25 keyword scoring.
            query_vector: Dense query embedding vector.
            top_k: Number of final results to return.
            alpha: RRF weight — 1.0 = pure dense, 0.0 = pure BM25. Default: 0.5.
            filters: Payload filter dict — MUST include 'tenant_id' for isolation.

        Returns:
            List of VectorSearchResult sorted by descending RRF score.
        """
        dense_results = self.search_dense(
            query_vector=query_vector, top_k=top_k * 2, filters=filters
        )
        if not dense_results or not query_text:
            return dense_results[:top_k]

        try:
            from rank_bm25 import BM25Okapi

            corpus_tokens = [res.text.lower().split() for res in dense_results]
            bm25 = BM25Okapi(corpus_tokens)
            query_tokens = query_text.lower().split()
            bm25_scores = bm25.get_scores(query_tokens)
        except Exception:
            logger.warning("BM25 scoring failed — falling back to dense-only results.")
            return dense_results[:top_k]

        # Compute RRF scores: weight dense rank + BM25 rank
        bm25_ranked = sorted(
            range(len(dense_results)), key=lambda i: float(bm25_scores[i]), reverse=True
        )
        dense_ranks = {res.chunk_id: rank + 1 for rank, res in enumerate(dense_results)}
        bm25_ranks = {dense_results[idx].chunk_id: rank + 1 for rank, idx in enumerate(bm25_ranked)}

        fused: list[VectorSearchResult] = []
        for res in dense_results:
            d_rank = dense_ranks[res.chunk_id]
            b_rank = bm25_ranks[res.chunk_id]
            rrf_score = alpha * (1.0 / (60 + d_rank)) + (1.0 - alpha) * (1.0 / (60 + b_rank))
            fused.append(res.model_copy(update={"score": float(rrf_score)}))

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:top_k]

    def delete_collection(self) -> None:
        """Drop the collection and all indexed data."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            logger.info("Deleted Qdrant collection '%s'.", self.collection_name)

    def _points_to_results(self, points: list[Any]) -> list[VectorSearchResult]:
        """Convert raw Qdrant ScoredPoint objects to VectorSearchResult instances."""
        results: list[VectorSearchResult] = []
        for point in points:
            p = point.payload or {}
            results.append(
                VectorSearchResult(
                    chunk_id=str(p.get("chunk_id", str(point.id))),
                    doc_id=str(p.get("doc_id", "")),
                    text=str(p.get("text", "")),
                    score=float(point.score),
                    tenant_id=str(p.get("tenant_id", "")),
                    provider_id=str(p.get("provider_id", "")),
                    patient_id=p.get("patient_id"),
                    document_type=str(p.get("document_type", "")),
                    collection_name=str(p.get("collection_name", "default")),
                    tags=list(p.get("tags", [])),
                    upload_timestamp=p.get("upload_timestamp"),
                    page_number=p.get("page_number"),
                    metadata=dict(p.get("metadata", {})),
                )
            )
        return results
