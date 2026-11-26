"""Standard locations for reproducible experiment artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_EXPERIMENT_NAME = re.compile(r"^[a-z0-9]+(?:[a-z0-9_-]*[a-z0-9]+)?$")


def experiment_name(config: dict[str, Any]) -> str:
    """Return the configured output directory name, validating its path safety."""
    name = str(config.get("experiment", {}).get("output_name", "baseline-v1")).strip()
    if not _EXPERIMENT_NAME.fullmatch(name):
        raise ValueError(
            "experiment.output_name must use lowercase letters, numbers, hyphens, or underscores"
        )
    return name


def results_root(config: dict[str, Any]) -> Path:
    """Return the root directory for committed result artifacts."""
    paths = config.get("paths", {})
    return Path(paths.get("results_dir", paths.get("output_dir", "results")))


def experiment_dir(config: dict[str, Any], mode: str = "centralized") -> Path:
    """Return the stable directory for one named experiment and execution mode."""
    return results_root(config) / "experiments" / mode / experiment_name(config)


def centralized_final_dir(config: dict[str, Any]) -> Path:
    """Return the directory for final centralized-training artifacts."""
    return experiment_dir(config, "centralized") / "final"


def centralized_tuning_dir(config: dict[str, Any]) -> Path:
    """Return the directory for Optuna artifacts of a centralized experiment."""
    return experiment_dir(config, "centralized") / "tuning"


__all__ = [
    "centralized_final_dir",
    "centralized_tuning_dir",
    "experiment_dir",
    "experiment_name",
    "results_root",
]
