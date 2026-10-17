"""Scaling utilities for feature and target data.

Both scalers are fitted only on the training split. Validation and test data
must be passed through :meth:`transform`, and model predictions can be
converted back to the original target scale with :meth:`inverse_transform`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import RobustScaler


class TrainOnlyScaler:
	"""Keep independent scalers for features and the target variable."""

	def __init__(self) -> None:
		self.feature_scaler = RobustScaler()
		self.target_scaler = RobustScaler()

	@staticmethod
	def _as_2d(data: Any) -> tuple[np.ndarray, bool]:
		array = np.asarray(data)
		was_1d = array.ndim == 1
		if was_1d:
			array = array.reshape(-1, 1)
		if array.ndim != 2:
			raise ValueError("Data must be one- or two-dimensional.")
		return array, was_1d

	@staticmethod
	def _restore_shape(data: np.ndarray, was_1d: bool) -> np.ndarray:
		return data.ravel() if was_1d else data

	def fit(self, x_train: Any, y_train: Any) -> "TrainOnlyScaler":
		"""Fit both scalers using training data only."""
		x_train_2d, _ = self._as_2d(x_train)
		y_train_2d, _ = self._as_2d(y_train)
		self.feature_scaler.fit(x_train_2d)
		self.target_scaler.fit(y_train_2d)
		return self

	def transform_features(self, x: Any) -> np.ndarray:
		"""Transform features with statistics learned from training data."""
		x_2d, was_1d = self._as_2d(x)
		return self._restore_shape(self.feature_scaler.transform(x_2d), was_1d)

	def transform_target(self, y: Any) -> np.ndarray:
		"""Transform a target with statistics learned from training data."""
		y_2d, was_1d = self._as_2d(y)
		return self._restore_shape(self.target_scaler.transform(y_2d), was_1d)

	def transform(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
		"""Transform features and target for validation or test data."""
		return self.transform_features(x), self.transform_target(y)

	def inverse_transform(self, prediction: Any) -> np.ndarray:
		"""Return predictions to the target's original scale."""
		prediction_2d, was_1d = self._as_2d(prediction)
		return self._restore_shape(
			self.target_scaler.inverse_transform(prediction_2d), was_1d
		)
