"""
AWS Bedrock Embedding Provider Implementation.

Uses direct boto3 bedrock-runtime API calls to execute embedding models:
- Amazon Titan v2 (supports Matryoshka dimension truncation: 1024, 512, 256)
- Cohere Embed v3 (supports input_type parameter: 'search_document' vs 'search_query')
"""

import json

import boto3
from tenacity import retry, stop_after_attempt, wait_exponential

from mentera_rag.embeddings.base import BaseEmbeddingProvider


class BedrockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Concrete EmbeddingProvider for AWS Bedrock models via raw boto3 client.
    """

    def __init__(
        self,
        model_name: str = "amazon.titan-embed-text-v2:0",
        dimension: int = 1024,
        region_name: str = "us-east-1",
        batch_size: int = 32,
    ):
        super().__init__(model_name=model_name, dimension=dimension)
        self.region_name = region_name
        self.batch_size = batch_size

        # Instantiate raw boto3 bedrock-runtime client
        self.client = boto3.client("bedrock-runtime", region_name=self.region_name)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _embed_titan(self, text: str) -> list[float]:
        """Invoke Amazon Titan Embeddings v2 with optional Matryoshka dimension truncation."""

        payload = {
            "inputText": text,
            "dimensions": self.dimension,
            "normalize": True,
        }
        response = self.client.invoke_model(
            modelId=self.model_name,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        response_body = json.loads(response.get("body", "{}").read())
        return list(response_body["embedding"])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _embed_cohere(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Invoke Cohere Embed v3 with explicit input_type."""
        payload = {
            "texts": texts,
            "input_type": input_type,
            "truncate": "END",
        }
        response = self.client.invoke_model(
            modelId=self.model_name,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response["body"].read())
        return [list(vec) for vec in response_body["embeddings"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents in batches."""
        if not texts:
            return []

        embeddings: list[list[float]] = []

        if "cohere" in self.model_name:
            # Batch process for Cohere v3
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i : i + self.batch_size]
                embeddings.extend(self._embed_cohere(chunk, input_type="search_document"))
        else:
            # Titan v2 processes single inputs per invoke_model API call
            for text in texts:
                embeddings.append(self._embed_titan(text))

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        if "cohere" in self.model_name:
            results = self._embed_cohere([text], input_type="search_query")
            return results[0]
        else:
            return self._embed_titan(text)
