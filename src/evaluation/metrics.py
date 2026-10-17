"""Regression metrics used by the project."""

import numpy as np


def _validate_inputs(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("y_true and y_pred must contain only finite values")
    return actual, predicted


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    actual, predicted = _validate_inputs(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    actual, predicted = _validate_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def calculate_metrics(y_true, y_pred):
    """Return the two project evaluation metrics."""
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}


__all__ = ["mae", "rmse", "calculate_metrics"]
