"""Evaluation helpers for validation and test forecasts."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


def _to_numpy(values: Any) -> np.ndarray:
	"""Convert NumPy, pandas, or tensor values to a flat float array."""
	if hasattr(values, "detach"):
		values = values.detach().cpu().numpy()
	values = np.asarray(values)
	if values.ndim == 0:
		return values.reshape(1).astype(float)
	if values.ndim > 1 and values.shape[-1] == 1:
		values = values.reshape(-1)
	return values.astype(float).reshape(-1)


def inverse_transform(values: Any, scaler: Any = None, target_index: int = 0) -> np.ndarray:
	"""Inverse-transform a target, including scalers fitted on many columns."""
	values = _to_numpy(values)
	if scaler is None:
		return values
	try:
		return _to_numpy(scaler.inverse_transform(values.reshape(-1, 1)))
	except (ValueError, TypeError):
		feature_count = len(getattr(scaler, "scale_", []))
		if not feature_count:
			raise
		matrix = np.zeros((len(values), feature_count), dtype=float)
		matrix[:, target_index] = values
		return np.asarray(scaler.inverse_transform(matrix))[:, target_index]


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
	valid = np.isfinite(actual) & np.isfinite(predicted)
	actual, predicted = actual[valid], predicted[valid]
	if actual.size == 0:
		return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
	error = predicted - actual
	return {
		"mae": float(np.mean(np.abs(error))),
		"rmse": float(np.sqrt(np.mean(error ** 2))),
	}


class Evaluator:
	"""Predict and evaluate validation/test data."""

	def __init__(self, scaler: Any = None, target_index: int = 0):
		self.scaler = scaler
		self.target_index = target_index

	def predict(
		self,
		model: Any,
		X: Any,
		y: Any = None,
		periods: Optional[Iterable[Any]] = None,
		split: Optional[str] = None,
	) -> pd.DataFrame:
		"""Create a table containing actual and predicted values."""
		predicted = inverse_transform(model.predict(X), self.scaler, self.target_index)
		data: dict[str, Any] = {"predicted": predicted}
		if y is not None:
			data["actual"] = inverse_transform(y, self.scaler, self.target_index)
		result = pd.DataFrame(data)
		if periods is not None:
			result["period"] = pd.to_datetime(list(periods))
		if split is not None:
			result["split"] = split
		return result

	def evaluate_split(
		self, model: Any, X: Any, y: Any,
		periods: Optional[Iterable[Any]] = None,
		split: Optional[str] = None,
	) -> tuple[pd.DataFrame, dict[str, float]]:
		result = self.predict(model, X, y, periods, split)
		return result, _metrics(result["actual"].to_numpy(), result["predicted"].to_numpy())

	def evaluate(
		self,
		model: Any,
		datasets: Mapping[str, tuple[Any, Any]],
		periods: Optional[Mapping[str, Iterable[Any]]] = None,
	) -> tuple[pd.DataFrame, pd.DataFrame]:
		"""Evaluate named datasets such as validation and test."""
		frames, summaries = [], []
		for name, (X, y) in datasets.items():
			frame, metrics = self.evaluate_split(
				model, X, y, None if periods is None else periods.get(name), name
			)
			frames.append(frame)
			summaries.append({"split": name, **metrics})
		return (
			pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
			pd.DataFrame(summaries),
		)

	@staticmethod
	def evaluate_by_period(results: pd.DataFrame, period: str = "period") -> pd.DataFrame:
		"""Calculate metrics for each period."""
		rows = []
		for value, group in results.groupby(period, sort=True):
			rows.append({period: value, **_metrics(group["actual"].to_numpy(), group["predicted"].to_numpy())})
		return pd.DataFrame(rows)

	@staticmethod
	def aggregate_annual(results: pd.DataFrame, period: str = "period") -> pd.DataFrame:
		"""Sum monthly actual and predicted values by calendar year."""
		frame = results.copy()
		frame[period] = pd.to_datetime(frame[period])
		frame["year"] = frame[period].dt.year
		return frame.groupby("year", as_index=False)[["actual", "predicted"]].sum(min_count=1)


def aggregate_annual(results: pd.DataFrame, period: str = "period") -> pd.DataFrame:
	"""Convenience wrapper for annual aggregation."""
	return Evaluator.aggregate_annual(results, period)
