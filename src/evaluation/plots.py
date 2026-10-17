"""Plotting utilities for evaluating forecasting models."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def _finish(save_path: Optional[str | Path] = None) -> None:
	"""Apply a common layout and optionally save the current figure."""
	plt.tight_layout()
	if save_path is not None:
		Path(save_path).parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_training_validation_loss(
	history: Mapping[str, Sequence[float]],
	save_path: Optional[str | Path] = None,
):
	"""Plot training and validation loss from a Keras-style history dict."""
	train = history.get("train_loss", history.get("loss", []))
	validation = history.get("val_loss", history.get("validation_loss", []))
	fig, ax = plt.subplots(figsize=(9, 5))
	ax.plot(range(1, len(train) + 1), train, label="Training loss", linewidth=2)
	if validation:
		ax.plot(
			range(1, len(validation) + 1),
			validation,
			label="Validation loss",
			linewidth=2,
		)
	ax.set(title="Training and Validation Loss", xlabel="Epoch", ylabel="Loss")
	ax.grid(alpha=0.3)
	ax.legend()
	_finish(save_path)
	return fig, ax


def plot_actual_vs_predicted(
	actual,
	predicted,
	dates=None,
	save_path: Optional[str | Path] = None,
):
	"""Plot actual values against predictions over time."""
	index = pd.to_datetime(dates) if dates is not None else range(len(actual))
	fig, ax = plt.subplots(figsize=(11, 5))
	ax.plot(index, actual, label="Actual", linewidth=2)
	ax.plot(index, predicted, label="Predicted", linewidth=2, alpha=0.85)
	ax.set(title="Actual versus Predicted", xlabel="Time", ylabel="Value")
	ax.grid(alpha=0.3)
	ax.legend()
	fig.autofmt_xdate()
	_finish(save_path)
	return fig, ax


def plot_error_over_time(
	actual,
	predicted,
	dates=None,
	save_path: Optional[str | Path] = None,
):
	"""Plot residual error (actual minus predicted) over time."""
	index = pd.to_datetime(dates) if dates is not None else range(len(actual))
	error = pd.Series(actual).reset_index(drop=True) - pd.Series(predicted).reset_index(
		drop=True
	)
	fig, ax = plt.subplots(figsize=(11, 5))
	ax.plot(index, error, color="tab:red", label="Error")
	ax.axhline(0, color="black", linestyle="--", linewidth=1)
	ax.set(title="Prediction Error over Time", xlabel="Time", ylabel="Actual - Predicted")
	ax.grid(alpha=0.3)
	ax.legend()
	fig.autofmt_xdate()
	_finish(save_path)
	return fig, ax


def plot_model_comparison(
	scores: Mapping[str, float] | pd.Series | pd.DataFrame,
	metric: str = "RMSE",
	save_path: Optional[str | Path] = None,
):
	"""Create a bar chart comparing one evaluation metric across models."""
	if isinstance(scores, pd.DataFrame):
		values = scores[metric]
	elif isinstance(scores, pd.Series):
		values = scores
	else:
		values = pd.Series(scores)
	values = values.sort_values()
	fig, ax = plt.subplots(figsize=(9, 5))
	values.plot.bar(ax=ax, color="tab:blue")
	ax.set(title=f"Model Comparison - {metric}", xlabel="Model", ylabel=metric)
	ax.grid(axis="y", alpha=0.3)
	_finish(save_path)
	return fig, ax


def plot_monthly_production(
	dates,
	actual,
	predicted,
	save_path: Optional[str | Path] = None,
):
	"""Compare actual and predicted production aggregated by month."""
	frame = pd.DataFrame({"date": pd.to_datetime(dates), "actual": actual, "predicted": predicted})
	monthly = frame.set_index("date").resample("MS").sum()
	fig, ax = plt.subplots(figsize=(11, 5))
	monthly.plot.bar(ax=ax, width=0.8, color=["tab:green", "tab:orange"])
	ax.set(title="Monthly Actual and Predicted Production", xlabel="Month", ylabel="Production")
	ax.grid(axis="y", alpha=0.3)
	ax.legend(["Actual", "Predicted"])
	fig.autofmt_xdate()
	_finish(save_path)
	return fig, ax
