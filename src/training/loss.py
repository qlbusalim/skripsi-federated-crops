"""Regression losses used by the experiment framework."""

import torch.nn as nn


def get_loss(name: str = "mse") -> nn.Module:
	"""Return the configured regression loss (MSE or MAE)."""
	choices = {
		"mse": nn.MSELoss,
		"mae": nn.L1Loss,
		"l1": nn.L1Loss,
	}
	key = name.lower().strip()
	if key not in choices:
		raise ValueError(f"Unsupported loss: {name}. Choose from mse or mae")
	return choices[key]()


MSELoss = nn.MSELoss
MAELoss = nn.L1Loss

__all__ = ["get_loss", "MSELoss", "MAELoss"]
