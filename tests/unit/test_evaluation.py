"""
Unit tests for Evaluation Framework and MLflow Tracker.
"""

from unittest.mock import MagicMock, patch

from mentera_rag.chunking.schemas import Chunk
from mentera_rag.evaluation.evaluator import RAGEvaluator
from mentera_rag.evaluation.metrics import (
    calculate_generation_metrics,
    calculate_ir_metrics,
    calculate_system_metrics,
)
from mentera_rag.evaluation.tracker import MLflowTracker


class TestEvaluationMetrics:
    """Tests for Layer 1, Layer 2, and Layer 3 metrics."""

    def test_calculate_ir_metrics(self):
        retrieved = ["doc_1", "doc_2", "doc_3"]
        ground_truth = ["doc_1"]

        metrics = calculate_ir_metrics(retrieved, ground_truth, k=10)

        assert metrics["mrr_at_10"] == 1.0
        assert metrics["recall_at_10"] == 1.0
        assert metrics["precision_at_10"] == 0.1
        assert metrics["ndcg_at_10"] == 1.0

    def test_calculate_generation_metrics(self):
        query = "What is PCD?"
        answer = "PCD is programmed cell death"
        context = "PCD is programmed cell death in organisms"

        metrics = calculate_generation_metrics(query, answer, context)

        assert metrics["faithfulness"] > 0.0
        assert metrics["answer_relevance"] > 0.0

    def test_calculate_system_metrics(self):
        metrics = calculate_system_metrics(
            retrieval_ms=10.0,
            rerank_ms=5.0,
            generation_ms=100.0,
            total_tokens=150,
        )

        assert metrics["total_latency_ms"] == 115.0
        assert metrics["total_tokens"] == 150


class TestMLflowTracker:
    """Tests for MLflowTracker."""

    @patch("mlflow.start_run")
    @patch("mlflow.set_experiment")
    @patch("mlflow.set_tracking_uri")
    def test_log_evaluation_run(self, mock_set_uri, mock_set_exp, mock_start_run):
        mock_run = MagicMock()
        mock_run.info.run_id = "mock-run-123"
        mock_start_run.return_value.__enter__.return_value = mock_run

        tracker = MLflowTracker(experiment_name="test_exp", tracking_uri="http://localhost:5000")
        run_id = tracker.log_evaluation_run(
            run_name="test_run",
            params={"vector_db": "qdrant"},
            metrics={"mrr_at_10": 0.85},
        )

        assert run_id == "mock-run-123"
        mock_set_uri.assert_called_once_with("http://localhost:5000")


class TestRAGEvaluator:
    """Tests for RAGEvaluator batch evaluator."""

    def test_evaluate_pipeline(self):
        mock_tracker = MagicMock()
        mock_tracker.log_evaluation_run.return_value = "run-abc-456"

        mock_pipeline = MagicMock()
        chunk = Chunk(
            chunk_id="c1",
            doc_id="pubmed_1",
            chunk_index=0,
            text="PCD context",
            token_count=2,
            tenant_id="tenant123",
            provider_id="provider456",
        )
        mock_pipeline.run.return_value = {
            "query": "What is PCD?",
            "answer": "PCD context answer",
            "retrieved_chunks": [chunk],
        }

        evaluator = RAGEvaluator(tracker=mock_tracker)
        test_cases = [{"query": "What is PCD?", "doc_id": "pubmed_1"}]
        result = evaluator.evaluate_pipeline(
            pipeline=mock_pipeline,
            test_cases=test_cases,
            run_name="eval_test_run",
            pipeline_params={"vector_db": "qdrant"},
        )

        assert result["run_id"] == "run-abc-456"
        assert "mrr_at_10" in result["metrics"]
        assert result["metrics"]["mrr_at_10"] == 1.0
