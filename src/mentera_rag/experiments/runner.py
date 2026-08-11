"""
Experiment Runner Implementation — Mentera RAG Pipeline.

Executes parameter sweeps combinations sequentially over Qdrant collections
and logs metrics (IR retrieval quality, indexing latencies) to MLflow.
"""

import time
import logging
from typing import Any

from mentera_rag.chunking.recursive import RecursiveCharacterChunker
from mentera_rag.chunking.schemas import Document as ChunkDocument
from mentera_rag.embeddings.factory import EmbeddingFactory
from mentera_rag.evaluation.evaluator import RAGEvaluator
from mentera_rag.evaluation.tracker import MLflowTracker
from mentera_rag.generation.base import BaseLLMProvider
from mentera_rag.orchestration.agentic import AgenticRAGGraph
from mentera_rag.orchestration.linear import LinearRAGPipeline
from mentera_rag.retrieval.bm25 import BM25Retriever
from mentera_rag.retrieval.dense import DenseRetriever
from mentera_rag.retrieval.ensemble import EnsembleRetriever
from mentera_rag.vector_stores.factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class MockLLMProvider(BaseLLMProvider):
    """Fallback LLM Provider for query rewriting in sweeps."""

    def __init__(self) -> None:
        super().__init__(model_name="mock-llm-v1")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return "optimized query text"


class ExperimentRunner:
    """
    Executes matrix evaluation sweeps and tracks runs in MLflow.
    """

    def __init__(self, tracker: MLflowTracker | None = None):
        self.tracker = tracker or MLflowTracker()
        self.evaluator = RAGEvaluator(tracker=self.tracker)

    def run_sweep(
        self,
        expanded_runs: list[dict[str, Any]],
        sample_docs: list[Any],
        test_cases: list[dict[str, Any]],
        experiment_name: str = "matrix_sweep",
    ) -> list[dict[str, Any]]:
        """
        Execute matrix sweep over expanded run configurations.
        """
        results: list[dict[str, Any]] = []

        for idx, run_params in enumerate(expanded_runs, start=1):
            orch_name = run_params.get("orchestration", "linear")
            run_name = f"run_{idx}_qdrant_{orch_name}"
            logger.info("Executing Sweep Run %d/%d: %s...", idx, len(expanded_runs), run_name)

            # 1. Chunking Setup
            chunk_size = run_params.get("chunking_chunk_size", 300)
            chunk_overlap = run_params.get("chunking_chunk_overlap", 30)
            chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            chunks = []
            parse_start = time.perf_counter()
            for doc in sample_docs:
                chunk_doc = ChunkDocument(
                    doc_id=doc.id,
                    content=doc.text,
                    source=doc.source,
                    tenant_id=run_params.get("tenant_id", "default_tenant"),
                    provider_id=run_params.get("provider_id", "default_provider"),
                    metadata=doc.metadata,
                )
                chunks.extend(chunker.chunk(chunk_doc))
            parse_time_ms = (time.perf_counter() - parse_start) * 1000.0

            # 2. Embedding Provider
            embed_provider_name = run_params.get("embedding_provider", "bedrock")
            embed_model = run_params.get("embedding_model", "amazon.titan-embed-text-v2:0")
            embed_provider = EmbeddingFactory.get_provider(
                provider_type=embed_provider_name,
                model_name=embed_model,
            )

            # 3. Embed chunks (measure embedding latency)
            chunk_texts = [c.text for c in chunks]
            embed_start = time.perf_counter()
            vectors = embed_provider.embed_documents(chunk_texts)
            embed_time_ms = (time.perf_counter() - embed_start) * 1000.0

            # 4. Vector Store (Index in Qdrant; measure indexing latency)
            vstore = VectorStoreFactory.get_vector_store(
                collection_name=f"m8_sweep_{idx}",
                dimension=embed_provider.dimension,
            )
            vstore.create_collection(force_recreate=True)

            index_start = time.perf_counter()
            vstore.index_chunks(chunks, vectors)
            index_time_ms = (time.perf_counter() - index_start) * 1000.0

            # 5. Retrieval Setup
            dense_ret = DenseRetriever(embed_provider=embed_provider, vector_store=vstore)
            bm25_ret = BM25Retriever(chunks=chunks)
            retriever = EnsembleRetriever(retrievers=[dense_ret, bm25_ret], rrf_k=60)

            # 6. LLM Provider (Query rewriter fallback)
            llm = MockLLMProvider()

            # 7. Orchestration Pipeline
            orchestration = run_params.get("orchestration", "linear")
            if orchestration == "agentic":
                pipeline: Any = AgenticRAGGraph(retriever=retriever, llm_provider=llm)
            else:
                pipeline = LinearRAGPipeline(retriever=retriever)

            # 8. Evaluate and Log to MLflow
            eval_res = self.evaluator.evaluate_pipeline(
                pipeline=pipeline,
                test_cases=test_cases,
                run_name=run_name,
                pipeline_params={
                    **run_params,
                    "parse_latency_ms": parse_time_ms,
                    "embed_latency_ms": embed_time_ms,
                    "index_latency_ms": index_time_ms,
                },
            )

            results.append(
                {
                    "run_name": run_name,
                    "run_id": eval_res["run_id"],
                    "params": run_params,
                    "metrics": {
                        **eval_res["metrics"],
                        "parse_time_ms": parse_time_ms,
                        "embed_time_ms": embed_time_ms,
                        "index_time_ms": index_time_ms,
                    },
                }
            )

        return results
