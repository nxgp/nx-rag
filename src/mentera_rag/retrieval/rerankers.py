"""
Reranker Provider Implementations (M5).

Provides Bedrock Rerank (boto3) and Local Cross-Encoder (sentence-transformers) rerankers.
"""

import json
from typing import Any

import boto3

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    from sentence_transformers import CrossEncoder

    HAS_CROSS_ENCODER = True
except ImportError:
    CrossEncoder = None  # type: ignore[misc]
    HAS_CROSS_ENCODER = False

from mentera_rag.chunking.schemas import Chunk
from mentera_rag.retrieval.base import BaseReranker


class BedrockReranker(BaseReranker):
    """
    Reranker using AWS Bedrock Rerank API via direct boto3 calls.
    """

    def __init__(
        self,
        model_name: str = "amazon.rerank-v1:0",
        region_name: str = "us-east-1",
    ):
        self.model_name = model_name
        self.region_name = region_name
        self.client = boto3.client("bedrock-runtime", region_name=self.region_name)

    def rerank(self, query: str, chunks: list[Chunk], top_n: int = 5) -> list[Chunk]:
        """Re-score candidate chunks using Bedrock Rerank API."""
        if not chunks:
            return []

        documents = [c.text for c in chunks]
        payload = {
            "queries": [{"text": query, "type": "TEXT"}],
            "documents": [{"text": doc, "type": "TEXT"} for doc in documents],
            "top_n": top_n,
        }

        response = self.client.invoke_model(
            modelId=self.model_name,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        body = json.loads(response["body"].read())
        results: list[dict[str, Any]] = body.get("results", [])

        reranked: list[Chunk] = []
        for res in results:
            idx = res.get("index", 0)
            if idx < len(chunks):
                reranked.append(chunks[idx])
        return reranked[:top_n]


class LocalReranker(BaseReranker):
    """
    Local Cross-Encoder reranker using sentence-transformers (BAAI/bge-reranker-v2-m3).
    Supports automatic GPU (CUDA) acceleration on available hardware.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
    ):
        self.model_name = model_name

        if not HAS_CROSS_ENCODER or CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required for LocalReranker. "
                "Install it with: pip install '.[local]'"
            )

        # Dynamically auto-detect GPU (cuda) if PyTorch/CUDA is available, otherwise fallback to CPU
        if device is not None:
            self.device = device
        elif HAS_TORCH and torch is not None and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(self, query: str, chunks: list[Chunk], top_n: int = 5) -> list[Chunk]:
        """Re-score query-passage pairs using local Cross-Encoder."""
        if not chunks:
            return []

        # Create pairs of (query, chunk_text)
        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)

        # Pair chunks with cross-encoder scores and sort
        scored_chunks = list(zip(chunks, scores, strict=False))
        scored_chunks.sort(key=lambda x: float(x[1]), reverse=True)

        return [chunk for chunk, score in scored_chunks[:top_n]]
