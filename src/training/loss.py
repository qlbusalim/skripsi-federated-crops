"""The regression loss used by the experiment framework."""

import torch.nn as nn


def get_loss(name: str = "mse") -> nn.Module:
	"""Return the one supported training loss: mean squared error."""
	if name.lower().strip() != "mse":
		raise ValueError("This project currently uses MSE loss only")
	return nn.MSELoss()


MSELoss = nn.MSELoss

__all__ = ["get_loss", "MSELoss"]
