"""Utilities for reproducible experiments."""

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
	"""Set the random seed for Python, NumPy, and PyTorch when available."""
	if not isinstance(seed, int):
		raise TypeError("seed must be an integer")

	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)

	try:
		import torch
	except ImportError:
		return

	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

	if deterministic:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


seed_everything = set_seed

__all__ = ["set_seed", "seed_everything"]
