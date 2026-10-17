"""Utilities for loading and preparing tabular time-series data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_COLUMNS = ("date", "region")


def load_data(
	filepath: str | Path,
	*,
	date_column: str = "date",
	region_column: str = "region",
	region: str | Iterable[str] | None = None,
	sort_by: str | list[str] | None = None,
	ascending: bool | list[bool] = True,
	**read_csv_kwargs,
) -> pd.DataFrame:
	"""Read a CSV and return a consistently typed, sorted DataFrame.

	``region`` can be a single region, an iterable of regions, or ``None``
	to keep all regions.  Dates are parsed with invalid values converted to
	``NaT`` and rows with invalid dates are discarded.
	"""
	path = Path(filepath)
	if not path.is_file():
		raise FileNotFoundError(f"CSV file not found: {path}")

	data = pd.read_csv(path, **read_csv_kwargs)
	required = {date_column, region_column}
	missing = required.difference(data.columns)
	if missing:
		names = ", ".join(sorted(missing))
		raise ValueError(f"Missing required column(s): {names}")

	data = data.copy()
	data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
	data = data.dropna(subset=[date_column])

	if region is not None:
		regions = [region] if isinstance(region, str) else list(region)
		data = data[data[region_column].isin(regions)]

	keys = sort_by or date_column
	data = data.sort_values(keys, ascending=ascending, kind="stable")
	return data.reset_index(drop=True)


def read_csv(*args, **kwargs) -> pd.DataFrame:
	"""Backward-compatible alias for :func:`load_data`."""
	return load_data(*args, **kwargs)


__all__ = ["DEFAULT_COLUMNS", "load_data", "read_csv"]