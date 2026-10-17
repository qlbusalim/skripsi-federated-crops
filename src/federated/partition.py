"""Partition provincial records into simulated federated clients."""

from collections.abc import Mapping
from typing import Any


def partition_by_province(data: Any, province_column: str = "province") -> dict[str, Any]:
    """Create one deterministic client partition for every province."""
    if hasattr(data, "columns") and hasattr(data, "loc"):
        if province_column not in data.columns:
            raise KeyError(f"Missing province column: {province_column}")
        provinces = sorted(data[province_column].dropna().unique())
        return {
            f"client_{index}": data.loc[data[province_column] == province].copy()
            for index, province in enumerate(provinces, 1)
        }

    rows = list(data)
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError("data must be a DataFrame or an iterable of mappings")
    provinces = sorted({row.get(province_column) for row in rows if row.get(province_column) is not None})
    return {
        f"client_{index}": [row for row in rows if row.get(province_column) == province]
        for index, province in enumerate(provinces, 1)
    }


partition_by_region = partition_by_province
partition_data = partition_by_province

__all__ = ["partition_by_province", "partition_by_region", "partition_data"]
