"""Central configuration loading for data and experiment settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/baseline.yaml") -> dict[str, Any]:
	"""Load a YAML configuration and validate its top-level structure."""
	source = Path(path)
	if not source.is_file():
		raise FileNotFoundError(f"Configuration file not found: {source}")
	with source.open("r", encoding="utf-8") as handle:
		config = yaml.safe_load(handle) or {}
	if not isinstance(config, dict):
		raise ValueError("Configuration root must be a mapping")
	return config


def get_config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
	"""Read a nested configuration value safely."""
	value: Any = config
	for key in keys:
		if not isinstance(value, dict) or key not in value:
			return default
		value = value[key]
	return value


__all__ = ["get_config_value", "load_config"]
