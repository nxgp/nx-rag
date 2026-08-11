"""
Comparison Reporter Implementation — Mentera RAG Pipeline.

Formats matrix sweep metrics into markdown tables and writes production recommendation reports.
"""

from pathlib import Path
from typing import Any


class ComparisonReporter:
    """
    Formats evaluation experiment metrics into markdown reports.
    """

    def __init__(self, output_dir: str | Path = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, run_results: list[dict[str, Any]]) -> str:
        """
        Generate markdown comparison report from experiment sweep results.
        """
        if not run_results:
            return "# Mentera RAG Experiment Comparison Report\n\nNo runs recorded."

        # Sort runs by NDCG@10 combined with Latency score
        # Note: Faithfulness is 0 since context-only pipelines omit generation
        sorted_runs = sorted(
            run_results,
            key=lambda r: (
                r["metrics"].get("ndcg_at_10", 0.0)
                - (r["metrics"].get("avg_latency_ms", 0.0) / 1000.0)
            ),
            reverse=True,
        )

        top_run = sorted_runs[0]

        report_lines = [
            "# Mentera RAG Experiment Comparison Report",
            "",
            "## Executive Summary & Production Recommendation",
            "",
            f"**Recommended Production Configuration**: `{top_run['run_name']}`",
            "- **Vector Store**: Qdrant",
            f"- **Embedding Model**: `{top_run['params'].get('embedding_model', 'N/A')}`",
            f"- **Orchestration**: `{top_run['params'].get('orchestration', 'N/A')}`",
            f"- **NDCG@10**: `{top_run['metrics'].get('ndcg_at_10', 0.0)}`",
            f"- **Avg Latency**: `{top_run['metrics'].get('avg_latency_ms', 0.0)} ms`",
            (
                f"- **Embed Ingestion Latency**: "
                f"`{round(top_run['metrics'].get('embed_time_ms', 0.0), 2)} ms`"
            ),
            "",
            "## Matrix Benchmark Comparison Table",
            "",
            "| Run Name | Embedding | Orchestration | MRR@10 | NDCG@10 | Avg Latency (ms) | Embed Time (ms) | Index Time (ms) |",  # noqa: E501
            "|---|---|---|---|---|---|---|---|",
        ]

        for r in sorted_runs:
            name = r["run_name"]
            emb = r["params"].get("embedding_model", "N/A").split("/")[-1]
            orch = r["params"].get("orchestration", "N/A")
            mrr = r["metrics"].get("mrr_at_10", 0.0)
            ndcg = r["metrics"].get("ndcg_at_10", 0.0)
            lat = r["metrics"].get("avg_latency_ms", 0.0)
            embed_t = round(r["metrics"].get("embed_time_ms", 0.0), 2)
            index_t = round(r["metrics"].get("index_time_ms", 0.0), 2)

            row = (
                f"| `{name}` | `{emb}` | `{orch}` | `{mrr}` | "
                f"`{ndcg}` | `{lat}` | `{embed_t}` | `{index_t}` |"
            )
            report_lines.append(row)

        report_content = "\n".join(report_lines) + "\n"

        report_file = self.output_dir / "comparison_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        return str(report_file)
