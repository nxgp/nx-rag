"""
FastAPI Production REST Service for Mentera RAG Pipeline.

Provides APIs for:
- health checks (/health)
- generating upload URLs (/upload/presign)
- triggering ingestion of uploaded files (/ingest)
- querying relevant context with tenant filters (/query)
- local storage PUT helper (/api/v1/upload/local)
"""

import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status

from mentera_rag.api.schemas import (
    IngestRequest,
    IngestResponse,
    PresignRequest,
    PresignResponse,
    QueryRequest,
    QueryResponse,
    RetrievedContext,
)
from mentera_rag.config.settings import settings
from mentera_rag.embeddings.factory import EmbeddingFactory
from mentera_rag.ingestion.upload_pipeline import UploadPipeline
from mentera_rag.storage.factory import StorageFactory
from mentera_rag.utils.logging import request_id_var, setup_logging
from mentera_rag.utils.rate_limit import TokenBucketRateLimiter
from mentera_rag.vector_stores.factory import VectorStoreFactory

# 1. Initialize structured logging configuration on import
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# 2. Instantiate Rate Limiters for sensitive endpoints
# Ingestion endpoints limits: 2.0 requests per second with capacity of 10.0 per tenant
presign_rate_limiter = TokenBucketRateLimiter(rate=2.0, capacity=10.0)
ingest_rate_limiter = TokenBucketRateLimiter(rate=1.0, capacity=5.0)

app = FastAPI(
    title="Mentera RAG Pipeline API",
    version="1.0.0",
    description="Production-grade, cloud-agnostic, multi-tenant RAG context retrieval service.",
)


@app.middleware("http")
async def request_id_and_logging_middleware(request: Request, call_next: Any) -> Response:
    """
    Middleware that assigns a unique request ID (UUID) per call, binds it
    to log context, and records request execution timing.
    """
    start_time = time.perf_counter()

    # Read existing request ID or generate a new one
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = request_id_var.set(request_id)

    logger.info(
        "Request started: %s %s",
        request.method,
        request.url.path,
        extra={"method": request.method, "path": request.url.path},
    )

    try:
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Inject request ID into response headers
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "Request finished: %s %s - Status %d - %dms",
            request.method,
            request.url.path,
            response.status_code,
            int(duration_ms),
            extra={
                "status_code": response.status_code,
                "latency_ms": round(duration_ms, 2),
            },
        )
        return response
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(
            "Request failed: %s %s - Exception %s - %dms",
            request.method,
            request.url.path,
            str(e),
            int(duration_ms),
            exc_info=True,
            extra={"latency_ms": round(duration_ms, 2)},
        )
        return Response(
            content=json.dumps({"detail": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )
    finally:
        # Reset context var
        request_id_var.reset(token)


@app.get("/health")
def health_check() -> dict[str, Any]:
    """
    Verify API health and status of connected resources (Qdrant & Storage).
    """
    qdrant_status = "healthy"
    storage_status = "healthy"

    # Check Qdrant connection
    try:
        from mentera_rag.vector_stores.qdrant_store import QdrantVectorStore

        v_store = VectorStoreFactory.get_vector_store()
        if isinstance(v_store, QdrantVectorStore):
            v_store.client.get_collections()
    except Exception as e:
        logger.error("Healthcheck: Qdrant connection failed: %s", e)
        qdrant_status = "unhealthy"

    # Check Storage connection
    try:
        s_store = StorageFactory.get_store()
        s_store.exists("health-check-nonexistent-key-123")
    except Exception as e:
        logger.error("Healthcheck: Storage connection failed: %s", e)
        storage_status = "unhealthy"

    overall_status = "healthy"
    if qdrant_status == "unhealthy" or storage_status == "unhealthy":
        overall_status = "degraded"

    return {
        "status": overall_status,
        "environment": settings.ENV,
        "services": {
            "qdrant": qdrant_status,
            "storage": storage_status,
        },
    }


@app.post("/upload/presign", response_model=PresignResponse)
def get_upload_presigned_url(request: PresignRequest) -> dict[str, Any]:
    """
    Endpoint 1: Generate a presigned PUT URL allowing clients to upload a file
    directly to the cloud storage provider (S3, Blob, GCS).
    """
    # Rate limit by tenant_id
    presign_rate_limiter.check(request.tenant_id)

    filename = request.filename.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    import os

    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext not in settings.ALLOWED_FILE_EXTENSIONS:
        allowed = settings.ALLOWED_FILE_EXTENSIONS
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {allowed}",
        )

    # Validate tenant format (alphanumeric, hyphens, underscores)
    import re

    if not re.match(r"^[a-zA-Z0-9\-_]+$", request.tenant_id):
        raise HTTPException(
            status_code=400,
            detail="tenant_id must be alphanumeric or contain only hyphens and underscores",
        )

    # Generate a unique storage key
    file_uuid = str(uuid.uuid4())
    storage_key = f"{request.tenant_id}/{request.provider_id}/{file_uuid}_{filename}"

    try:
        store = StorageFactory.get_store()
        url = store.generate_presigned_upload_url(
            key=storage_key,
            content_type=request.content_type,
            expires_in=3600,
        )
        return {
            "upload_url": url,
            "storage_key": storage_key,
            "expires_in": 3600,
        }
    except Exception as e:
        logger.error("Failed to generate upload URL: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Internal storage error generating upload parameters.",
        ) from e


@app.post("/ingest", response_model=IngestResponse)
def trigger_file_ingestion(request: IngestRequest) -> dict[str, Any]:
    """
    Endpoint 2: Trigger the ingestion pipeline (parse, chunk, embed, index)
    for a file that has been successfully uploaded to the object store.
    """
    # Rate limit by tenant_id
    ingest_rate_limiter.check(request.tenant_id)

    try:
        pipeline = UploadPipeline()
        result = pipeline.run(
            storage_key=request.storage_key,
            tenant_id=request.tenant_id,
            provider_id=request.provider_id,
            patient_id=request.patient_id,
            tags=request.tags,
            collection_name=request.collection_name,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Source file not found in storage. Ensure upload completed. Error: {e}",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process and index document. Error: {e}",
        ) from e


@app.post("/query", response_model=QueryResponse)
def query_context(request: QueryRequest) -> dict[str, Any]:
    """
    Endpoint 3: Retrieve top-K relevant text passages using tenant-level payload filters.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")

    start_time = time.perf_counter()

    try:
        # 1. Initialize vector store & embed query
        vector_store = VectorStoreFactory.get_vector_store()
        embedding_provider = EmbeddingFactory.get_provider()

        query_vector = embedding_provider.embed_query(request.query)

        # 2. Build tenant isolation filters
        # tenant_id and provider_id are mandatory
        filters: dict[str, Any] = {
            "tenant_id": request.tenant_id,
            "provider_id": request.provider_id,
        }

        # Optional filters
        if request.patient_id:
            filters["patient_id"] = request.patient_id
        if request.collection_name:
            filters["collection_name"] = request.collection_name

        # 3. Perform Qdrant hybrid search
        search_results = vector_store.search_hybrid(
            query_text=request.query,
            query_vector=query_vector,
            top_k=request.top_k,
            alpha=0.5,
            filters=filters,
        )

        # Filter by tags if list provided
        if request.tags:
            search_results = [
                res for res in search_results if any(tag in res.tags for tag in request.tags)
            ]

        # 4. Map to RetrievedContext schemas
        retrieved_contexts = [
            RetrievedContext(
                chunk_id=res.chunk_id,
                doc_id=res.doc_id,
                text=res.text,
                score=res.score,
                metadata={
                    "page_number": res.page_number,
                    "document_type": res.document_type,
                    "tags": res.tags,
                    "upload_timestamp": str(res.upload_timestamp) if res.upload_timestamp else None,
                    **res.metadata,
                },
            )
            for res in search_results
        ]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": "success",
            "query": request.query,
            "retrieved_contexts": retrieved_contexts,
            "total_results": len(retrieved_contexts),
            "latency_ms": round(elapsed_ms, 2),
        }

    except Exception as e:
        logger.error("Query retrieval failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Context retrieval error: {e}",
        ) from e


@app.put("/api/v1/upload/local", status_code=status.HTTP_201_CREATED)
async def upload_local_file(request: Request, key: str) -> dict[str, Any]:
    """
    Development PUT Helper: Accept raw file upload bytes and store them
    in the local simulated storage directory at the resolved key path.
    """
    try:
        from mentera_rag.storage.local import LocalStorageStore

        if not key:
            raise HTTPException(status_code=400, detail="Storage key must be specified")

        body = await request.body()
        store = StorageFactory.get_store(provider_type="local")
        if not isinstance(store, LocalStorageStore):
            raise HTTPException(
                status_code=400,
                detail="Local upload endpoint is only active when using STORAGE_PROVIDER='local'",
            )

        store.upload_file(key, body)
        return {"status": "success", "storage_key": key, "bytes_written": len(body)}
    except Exception as e:
        logger.error("Local PUT helper upload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
