"""
RAG Pipeline Evaluator.

Executes batch evaluation over dataset records and reports metrics to MLflow.
"""

import time
from typing import Any

from mentera_rag.evaluation.metrics import (
    calculate_generation_metrics,
    calculate_ir_metrics,
    calculate_system_metrics,
)
from mentera_rag.evaluation.tracker import MLflowTracker


class RAGEvaluator:
    """
    Evaluates end-to-end RAG pipelines and tracks experiments via MLflowTracker.
    """

    def __init__(self, tracker: MLflowTracker | None = None):
        self.tracker = tracker or MLflowTracker()

    def evaluate_pipeline(
        self,
        pipeline: Any,
        test_cases: list[dict[str, Any]],
        run_name: str,
        pipeline_params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Evaluate a pipeline instance across test cases and log to MLflow.
        """
        aggregated_metrics: dict[str, float] = {
            "mrr_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "avg_latency_ms": 0.0,
        }

        total_runs = len(test_cases)
        if total_runs == 0:
            return {"run_id": "", "metrics": aggregated_metrics}

        total_mrr = 0.0
        total_ndcg = 0.0
        total_faithfulness = 0.0
        total_relevance = 0.0
        total_latency = 0.0

        for case in test_cases:
            query = case["query"]
            ground_truth_doc_id = case.get("doc_id", "")

            start_time = time.perf_counter()
            result = pipeline.run(query)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000.0
            retrieved_chunks = result.get("retrieved_chunks", [])
            answer = result.get("answer", "")

            # 1. IR Metrics
            retrieved_ids = [c.doc_id for c in retrieved_chunks]
            ir = calculate_ir_metrics(retrieved_ids, [ground_truth_doc_id], k=10)
            total_mrr += ir.get("mrr_at_10", 0.0)
            total_ndcg += ir.get("ndcg_at_10", 0.0)

            # 2. Generation Metrics
            context_str = " ".join([c.text for c in retrieved_chunks])
            gen = calculate_generation_metrics(query, answer, context_str)
            total_faithfulness += gen.get("faithfulness", 0.0)
            total_relevance += gen.get("answer_relevance", 0.0)

            # 3. System Latency
            sys_m = calculate_system_metrics(
                retrieval_ms=latency_ms * 0.4,
                rerank_ms=latency_ms * 0.1,
                generation_ms=latency_ms * 0.5,
            )
            total_latency += sys_m.get("total_latency_ms", 0.0)

        # Average metrics across test batch
        aggregated_metrics = {
            "mrr_at_10": round(total_mrr / total_runs, 4),
            "ndcg_at_10": round(total_ndcg / total_runs, 4),
            "faithfulness": round(total_faithfulness / total_runs, 4),
            "answer_relevance": round(total_relevance / total_runs, 4),
            "avg_latency_ms": round(total_latency / total_runs, 2),
        }

        # Log run to MLflow
        run_id = self.tracker.log_evaluation_run(
            run_name=run_name,
            params=pipeline_params,
            metrics=aggregated_metrics,
        )

        return {
            "run_id": run_id,
            "metrics": aggregated_metrics,
        }
