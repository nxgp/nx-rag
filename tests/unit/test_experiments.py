"""
Unit tests for Experiment Runner, Matrix Expander, and Comparison Reporter.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mentera_rag.experiments.matrix import MatrixExpander
from mentera_rag.experiments.reporter import ComparisonReporter
from mentera_rag.experiments.runner import ExperimentRunner


class TestMatrixExpander:
    """Tests for MatrixExpander."""

    def test_matrix_expansion(self, tmp_path):
        yaml_file = tmp_path / "test_exp.yaml"
        yaml_file.write_text(
            """
experiment:
  name: test_matrix
matrix:
  embedding_provider: [bedrock, azure]
  orchestration: [linear, agentic]
"""
        )

        expander = MatrixExpander(config_path=yaml_file)
        runs = expander.expand()

        assert len(runs) == 4
        run_combos = {(r["embedding_provider"], r["orchestration"]) for r in runs}
        assert run_combos == {
            ("bedrock", "linear"),
            ("bedrock", "agentic"),
            ("azure", "linear"),
            ("azure", "agentic"),
        }


class TestComparisonReporter:
    """Tests for ComparisonReporter."""

    def test_generate_report(self, tmp_path):
        reporter = ComparisonReporter(output_dir=tmp_path)
        run_results = [
            {
                "run_name": "run_1_bedrock",
                "params": {
                    "embedding_provider": "bedrock",
                    "embedding_model": "titan",
                    "orchestration": "linear",
                },
                "metrics": {
                    "ndcg_at_10": 0.85,
                    "avg_latency_ms": 300.0,
                    "embed_time_ms": 50.0,
                    "index_time_ms": 10.0,
                },
            },
            {
                "run_name": "run_2_azure",
                "params": {
                    "embedding_provider": "azure",
                    "embedding_model": "text-embedding-3-small",
                    "orchestration": "agentic",
                },
                "metrics": {
                    "ndcg_at_10": 0.90,
                    "avg_latency_ms": 450.0,
                    "embed_time_ms": 60.0,
                    "index_time_ms": 12.0,
                },
            },
        ]

        report_file = reporter.generate_report(run_results)
        assert Path(report_file).exists()

        content = (tmp_path / "comparison_report.md").read_text()
        assert "Mentera RAG Experiment Comparison Report" in content
        assert "run_2_azure" in content
        assert "Embed Time (ms)" in content


class TestExperimentRunner:
    """Tests for ExperimentRunner."""

    @patch("mentera_rag.vector_stores.qdrant_store.QdrantClient")
    @patch("mentera_rag.experiments.runner.RAGEvaluator")
    @patch("mentera_rag.experiments.runner.EmbeddingFactory.get_provider")
    def test_run_sweep(self, mock_get_provider, mock_evaluator_class, mock_qdrant_class):
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.collection_exists.return_value = True
        mock_qdrant_class.return_value = mock_qdrant_client

        mock_provider = MagicMock()
        mock_provider.dimension = 256
        mock_provider.embed_documents.return_value = [[0.1, 0.2]]
        mock_provider.model_name = "titan"
        mock_get_provider.return_value = mock_provider

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_pipeline.return_value = {
            "run_id": "run-123",
            "metrics": {"ndcg_at_10": 0.85, "avg_latency_ms": 250.0},
        }
        mock_evaluator_class.return_value = mock_evaluator

        mock_tracker = MagicMock()
        runner = ExperimentRunner(tracker=mock_tracker)
        expanded_runs = [{"embedding_provider": "bedrock", "orchestration": "linear"}]
        mock_doc = MagicMock()
        mock_doc.id = "doc1"
        mock_doc.text = "Context"
        mock_doc.source = "pubmed"
        mock_doc.metadata = {}

        results = runner.run_sweep(
            expanded_runs=expanded_runs,
            sample_docs=[mock_doc],
            test_cases=[{"query": "PCD?", "doc_id": "doc1"}],
        )

        assert len(results) == 1
        assert results[0]["run_name"] == "run_1_qdrant_linear"
        assert results[0]["metrics"]["ndcg_at_10"] == 0.85
        assert "embed_time_ms" in results[0]["metrics"]
