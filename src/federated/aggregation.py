"""Federated learning aggregation utilities."""

from collections import OrderedDict
from typing import Mapping, Sequence

import torch


def fedavg(
	client_state_dicts: Sequence[Mapping[str, torch.Tensor]],
	client_sizes: Sequence[int] | None = None,
) -> OrderedDict[str, torch.Tensor]:
	"""Return the weighted average of client model state dictionaries.

	When ``client_sizes`` is provided, each client's contribution is
	proportional to its number of training samples. Otherwise, clients are
	weighted equally.
	"""
	if not client_state_dicts:
		raise ValueError("client_state_dicts must not be empty")

	number_of_clients = len(client_state_dicts)
	if client_sizes is None:
		weights = [1.0] * number_of_clients
	else:
		if len(client_sizes) != number_of_clients:
			raise ValueError("client_sizes must match the number of clients")
		if any(size < 0 for size in client_sizes):
			raise ValueError("client_sizes must be non-negative")
		weights = [float(size) for size in client_sizes]

	total_weight = sum(weights)
	if total_weight <= 0:
		raise ValueError("the total client weight must be greater than zero")

	keys = tuple(client_state_dicts[0].keys())
	if any(tuple(state.keys()) != keys for state in client_state_dicts[1:]):
		raise ValueError("all client state dictionaries must have the same keys")

	aggregated = OrderedDict()
	for key in keys:
		value = client_state_dicts[0][key]
		if not isinstance(value, torch.Tensor):
			aggregated[key] = value
			continue

		if value.is_floating_point() or value.is_complex():
			result = torch.zeros_like(value)
			for state, weight in zip(client_state_dicts, weights):
				result.add_(state[key], alpha=weight / total_weight)
			aggregated[key] = result
		else:
			# Integer buffers (for example BatchNorm counters) cannot be
			# averaged; retain the first client's value.
			aggregated[key] = value.clone()

	return aggregated


def weighted_fedavg(
	client_state_dicts: Sequence[Mapping[str, torch.Tensor]],
	client_weights: Sequence[float],
) -> OrderedDict[str, torch.Tensor]:
	"""Convenience alias for FedAvg with explicit client weights."""
	if len(client_weights) != len(client_state_dicts):
		raise ValueError("client_weights must match the number of clients")
	if any(weight < 0 for weight in client_weights):
		raise ValueError("client_weights must be non-negative")
	return fedavg(client_state_dicts, client_sizes=client_weights)
