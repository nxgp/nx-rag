"""
MLflow Experiment Tracker.

Manages experiment creation, parameter logging, metric tracking, and artifact logging.
"""

from typing import Any

import mlflow

from mentera_rag.config.settings import settings


class MLflowTracker:
    """
    MLflow tracking helper for logging pipeline benchmarks.
    """

    def __init__(
        self,
        experiment_name: str = "mentera_rag_evaluation",
        tracking_uri: str | None = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def log_evaluation_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: list[str] | None = None,
    ) -> str:
        """
        Log an evaluation run to MLflow server.

        Args:
            run_name: Human-readable run name (e.g. 'qdrant_bge_hybrid_linear').
            params: Dictionary of configuration parameters.
            metrics: Dictionary of numerical evaluation scores.
            artifacts: Optional list of file paths to attach to MLflow run.

        Returns:
            MLflow run ID string.
        """
        with mlflow.start_run(run_name=run_name) as run:
            # 1. Log configuration parameters
            mlflow.log_params(params)

            # 2. Log numerical metrics
            mlflow.log_metrics(metrics)

            # 3. Log artifacts (reports/plots)
            if artifacts:
                for file_path in artifacts:
                    mlflow.log_artifact(file_path)

            return str(run.info.run_id)
