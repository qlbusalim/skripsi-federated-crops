"""Deterministic feature engineering for time-indexed experiment inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def add_configured_features(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Return a copy of *data* with feature engineering enabled by config.

    Sinusoidal calendar features are computed from the timestamp itself, so no
    training statistics are learned and no future target information is used.
    """
    data_config = config.get("data", {})
    date_column = data_config.get("date_column", "date")
    engineering = data_config.get("feature_engineering", {})
    sinusoidal = engineering.get("sinusoidal", {})
    if not sinusoidal.get("enabled", False):
        return data

    if date_column not in data.columns:
        raise KeyError(f"Cannot build sinusoidal features: missing '{date_column}'")
    period = int(sinusoidal.get("period", 12))
    if period < 2:
        raise ValueError("data.feature_engineering.sinusoidal.period must be at least 2")

    result = data.copy()
    dates = pd.to_datetime(result[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Cannot build sinusoidal features: invalid values in '{date_column}'")
    if period == 12 and sinusoidal.get("source", "month") == "month":
        phase = dates.dt.month.to_numpy(dtype=float) - 1.0
    else:
        raise ValueError("Only monthly sinusoidal encoding is currently supported")

    angle = 2.0 * np.pi * phase / period
    result["month_sin"] = np.sin(angle)
    result["month_cos"] = np.cos(angle)
    return result


__all__ = ["add_configured_features"]
