"""Simple forecasting baselines for small datasets."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


def _validate_series(values):
	values = np.asarray(values, dtype=float).reshape(-1)
	if values.size == 0:
		raise ValueError("values must not be empty")
	if not np.isfinite(values).all():
		raise ValueError("values must contain only finite numbers")
	return values


def _validate_steps(steps):
	if int(steps) != steps or steps < 1:
		raise ValueError("steps must be a positive integer")
	return int(steps)


class NaiveOneStep:
	"""Forecast every future value as the most recent observed value."""

	def fit(self, values):
		self.last_value_ = _validate_series(values)[-1]
		return self

	def predict(self, steps=1):
		if not hasattr(self, "last_value_"):
			raise RuntimeError("fit must be called before predict")
		return np.repeat(self.last_value_, _validate_steps(steps))


class SeasonalNaive:
	"""Repeat the last complete seasonal cycle."""

	def __init__(self, season_length):
		if int(season_length) != season_length or season_length < 1:
			raise ValueError("season_length must be a positive integer")
		self.season_length = int(season_length)

	def fit(self, values):
		values = _validate_series(values)
		if len(values) < self.season_length:
			raise ValueError("at least one complete season is required")
		self.season_ = values[-self.season_length :].copy()
		return self

	def predict(self, steps=1):
		if not hasattr(self, "season_"):
			raise RuntimeError("fit must be called before predict")
		return np.resize(self.season_, _validate_steps(steps))


class _AutoregressiveRegression:
	estimator = LinearRegression

	def __init__(self, n_lags=1, **kwargs):
		if int(n_lags) != n_lags or n_lags < 1:
			raise ValueError("n_lags must be a positive integer")
		self.n_lags = int(n_lags)
		self.kwargs = kwargs

	def fit(self, values):
		values = _validate_series(values)
		if len(values) <= self.n_lags:
			raise ValueError("more observations than n_lags are required")
		X = np.array(
			[values[i - self.n_lags : i] for i in range(self.n_lags, len(values))]
		)
		self.model_ = self.estimator(**self.kwargs).fit(X, values[self.n_lags :])
		self.history_ = values.tolist()
		return self

	def predict(self, steps=1):
		if not hasattr(self, "model_"):
			raise RuntimeError("fit must be called before predict")
		history = self.history_.copy()
		forecast = []
		for _ in range(_validate_steps(steps)):
			x = np.asarray(history[-self.n_lags :]).reshape(1, -1)
			value = float(self.model_.predict(x)[0])
			forecast.append(value)
			history.append(value)
		return np.asarray(forecast)


class LinearRegressionBaseline(_AutoregressiveRegression):
	"""Autoregressive ordinary least-squares baseline."""

	estimator = LinearRegression


class RidgeRegressionBaseline(_AutoregressiveRegression):
	"""Autoregressive L2-regularized linear baseline."""

	estimator = Ridge

	def __init__(self, n_lags=1, alpha=1.0, **kwargs):
		super().__init__(n_lags=n_lags, alpha=alpha, **kwargs)


# Convenient aliases for experiment scripts.
Naive = NaiveOneStep
LinearBaseline = LinearRegressionBaseline
RidgeBaseline = RidgeRegressionBaseline


__all__ = [
	"NaiveOneStep",
	"Naive",
	"SeasonalNaive",
	"LinearRegressionBaseline",
	"LinearBaseline",
	"RidgeRegressionBaseline",
	"RidgeBaseline",
]
