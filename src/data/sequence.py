"""Sliding-window construction for grouped time-series data."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def make_sequences(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    window_size: int = 12,
    horizon: int = 1,
    group_column: str | None = "province",
    date_column: str = "date",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create non-overlapping-group-safe forecasting windows.

    Windows are built independently for every group, so a sequence cannot
    accidentally continue from one province into another. This helper does
    not perform train/validation/test splitting.
    """
    if window_size < 1 or horizon < 1:
        raise ValueError("window_size and horizon must be positive")
    required = set(feature_columns) | {target_column, date_column}
    if group_column:
        required.add(group_column)
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing sequence columns: {', '.join(sorted(missing))}")

    group_values = data[group_column].drop_duplicates().tolist() if group_column else [None]
    windows, targets, metadata = [], [], []
    for group in group_values:
        frame = data if group is None else data[data[group_column] == group]
        frame = frame.sort_values(date_column, kind="stable").reset_index(drop=True)
        values = frame[list(feature_columns)].to_numpy(dtype=float)
        target = frame[target_column].to_numpy(dtype=float)
        limit = len(frame) - window_size - horizon + 1
        for start in range(max(0, limit)):
            target_index = start + window_size + horizon - 1
            windows.append(values[start : start + window_size])
            targets.append(target[target_index])
            metadata.append({
                date_column: frame.loc[target_index, date_column],
                **({group_column: group} if group_column else {}),
            })

    if not windows:
        raise ValueError("Not enough observations to create a sequence")
    return np.asarray(windows), np.asarray(targets, dtype=float).reshape(-1, 1), pd.DataFrame(metadata)


__all__ = ["make_sequences"]
