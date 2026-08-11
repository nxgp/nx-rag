"""
Experiment Matrix Expander.

Parses YAML experiment matrix definitions and computes Cartesian product parameter sweeps.
"""

import itertools
from pathlib import Path
from typing import Any

import yaml


class MatrixExpander:
    """
    Expands declarative YAML matrix configs into executable run combination dicts.
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.raw_config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Read and parse YAML experiment config file with example fallback."""
        target_path = self.config_path
        if not target_path.exists():
            example_path = target_path.parent / "experiment.example.yaml"
            if example_path.exists():
                target_path = example_path
            else:
                raise FileNotFoundError(f"Experiment config file not found at: {self.config_path}")

        with open(target_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return dict(data)

    def expand(self) -> list[dict[str, Any]]:
        """
        Compute Cartesian product of all matrix axes.

        Returns:
            List of flattened configuration dictionaries for each experiment run.
        """
        matrix_cfg: dict[str, Any] = self.raw_config.get("matrix", {})
        if not matrix_cfg:
            return []

        keys = list(matrix_cfg.keys())
        values_lists: list[list[Any]] = []

        for key in keys:
            val = matrix_cfg[key]
            if isinstance(val, list):
                values_lists.append(val)
            else:
                values_lists.append([val])

        # Compute Cartesian Product across all axes
        expanded_runs: list[dict[str, Any]] = []
        for combo in itertools.product(*values_lists):
            run_params: dict[str, Any] = {}
            for k, v in zip(keys, combo, strict=False):
                if isinstance(v, dict):
                    # Flatten nested dictionary parameters (e.g. chunking, retrieval)
                    for sub_k, sub_v in v.items():
                        run_params[f"{k}_{sub_k}"] = sub_v
                else:
                    run_params[k] = v
            expanded_runs.append(run_params)

        return expanded_runs
