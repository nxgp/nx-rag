"""
Dense Vector Retriever Implementation (M5).

Translates query text into a vector using EmbeddingProvider (M3) and searches
nearest neighbors in BaseVectorStore (M4).
"""

from typing import Any

from mentera_rag.chunking.schemas import Chunk
from mentera_rag.embeddings.base import BaseEmbeddingProvider
from mentera_rag.retrieval.base import BaseRetriever
from mentera_rag.vector_stores.base import BaseVectorStore


class DenseRetriever(BaseRetriever):
    """
    Dense semantic retriever executing vector similarity search against Qdrant or Weaviate.
    """

    def __init__(
        self,
        embed_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
    ):
        self.embed_provider = embed_provider
        self.vector_store = vector_store

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[Chunk]:
        """Embed query and search vector store nearest neighbors."""
        # Step 1: Embed query into a dense vector
        query_vector = self.embed_provider.embed_query(query)

        # Step 2: Search vector database
        search_results = self.vector_store.search_dense(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )

        # Step 3: Map VectorSearchResult items back to Chunk objects
        chunks: list[Chunk] = []
        for i, res in enumerate(search_results or []):
            c = Chunk(
                chunk_id=res.chunk_id,
                doc_id=res.doc_id,
                chunk_index=i,
                text=res.text,
                tenant_id=res.tenant_id,
                provider_id=res.provider_id,
                patient_id=res.patient_id,
                document_type=res.document_type,
                collection_name=res.collection_name,
                tags=res.tags,
                upload_timestamp=res.upload_timestamp,
                page_number=res.page_number,
                token_count=len(res.text.split()),
                score=getattr(res, "score", 0.0) or 0.0,
                metadata=res.metadata,
            )
            chunks.append(c)
        return chunks
