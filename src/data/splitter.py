"""Time-based dataset splitting utilities.

Splitting is performed before fitting a scaler so that validation/test data
cannot leak into the training statistics.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def split_by_time(
	data: pd.DataFrame,
	time_column: str,
	test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Sort *data* by time and return chronological train/test partitions."""
	if not isinstance(data, pd.DataFrame):
		raise TypeError("data must be a pandas.DataFrame")
	if time_column not in data.columns:
		raise KeyError(f"Column not found: {time_column}")
	if not 0 < test_size < 1:
		raise ValueError("test_size must be between 0 and 1")
	if len(data) < 2:
		raise ValueError("data must contain at least two rows")

	ordered = data.sort_values(time_column, kind="stable").reset_index(drop=True)
	split_index = max(1, min(int(len(ordered) * (1 - test_size)), len(ordered) - 1))
	return (
		ordered.iloc[:split_index].copy(),
		ordered.iloc[split_index:].copy(),
	)


def split_and_scale(
	data: pd.DataFrame,
	time_column: str,
	target_column: str | None = None,
	test_size: float = 0.2,
	scaler: Any = None,
) -> tuple[Any, Any, Any, Any, Any]:
	"""Split by time, then fit ``scaler`` using training features only.

	Returns ``(X_train, X_test, y_train, y_test, scaler)``.  The time column
	is excluded from the features and the scaler is never fitted on test data.
	"""
	train, test = split_by_time(data, time_column, test_size)

	if target_column is not None:
		if target_column not in data.columns:
			raise KeyError(f"Column not found: {target_column}")
		y_train = train.pop(target_column)
		y_test = test.pop(target_column)
	else:
		y_train = y_test = None

	X_train = train.drop(columns=time_column)
	X_test = test.drop(columns=time_column)

	if scaler is not None:
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)

	return X_train, X_test, y_train, y_test, scaler
