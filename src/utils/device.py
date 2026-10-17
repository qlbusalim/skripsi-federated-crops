"""Utilities for selecting and working with the computation device."""

from __future__ import annotations

import torch


def get_device(preferred: str | None = None) -> torch.device:
	"""Return the best available device, optionally honoring a preference."""
	if preferred:
		requested = torch.device(preferred)
		if requested.type == "cuda" and not torch.cuda.is_available():
			return torch.device("cpu")
		if requested.type == "mps" and not (
			hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
		):
			return torch.device("cpu")
		return requested

	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


DEVICE = get_device()


def move_to_device(value, device: torch.device | str = DEVICE):
	"""Recursively move tensors in common containers to ``device``."""
	if isinstance(value, torch.Tensor):
		return value.to(device)
	if isinstance(value, dict):
		return {key: move_to_device(item, device) for key, item in value.items()}
	if isinstance(value, list):
		return [move_to_device(item, device) for item in value]
	if isinstance(value, tuple):
		return tuple(move_to_device(item, device) for item in value)
	return value
