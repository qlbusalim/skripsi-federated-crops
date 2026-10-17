"""Coordination logic for federated learning rounds."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


Parameters = Sequence[np.ndarray]


@dataclass
class ClientUpdate:
	"""Result returned by a client after local training."""

	parameters: Parameters
	num_examples: int
	metrics: Mapping[str, float] | None = None


class FederatedServer:
	"""Initialize, distribute, aggregate, and evaluate a global model.

	The supplied model must implement ``get_parameters`` and
	``set_parameters``. Clients must implement ``fit(parameters, round_number)``
	and may implement ``evaluate(parameters, round_number)``.
	"""

	def __init__(
		self,
		model: Any,
		clients: Iterable[Any],
		rounds: int,
		fraction_fit: float = 1.0,
		fraction_evaluate: float = 1.0,
		evaluate_fn: Callable[[Parameters, int], Mapping[str, Any]] | None = None,
		seed: int | None = None,
	) -> None:
		if rounds < 1 or not 0 < fraction_fit <= 1 or not 0 < fraction_evaluate <= 1:
			raise ValueError("invalid rounds or client sampling fraction")
		self.model = model
		self.clients = list(clients)
		if not self.clients:
			raise ValueError("at least one client is required")
		self.rounds = rounds
		self.fraction_fit = fraction_fit
		self.fraction_evaluate = fraction_evaluate
		self.evaluate_fn = evaluate_fn
		self._random = random.Random(seed)
		self.global_parameters: list[np.ndarray] | None = None

	def initialize_global_model(self) -> list[np.ndarray]:
		"""Read and store the initial parameters from the global model."""
		self.global_parameters = self._copy(self.model.get_parameters())
		return self._copy(self.global_parameters)

	def send_model(self, client: Any, round_number: int) -> ClientUpdate:
		"""Send a copy of the global model to one client for local training."""
		if self.global_parameters is None:
			raise RuntimeError("global model is not initialized")
		result = client.fit(self._copy(self.global_parameters), {"round": round_number})
		if isinstance(result, ClientUpdate):
			return result
		parameters, num_examples, *rest = result
		return ClientUpdate(
			self._copy(parameters), int(num_examples), rest[0] if rest else None
		)

	def collect_parameters(self, round_number: int) -> list[ClientUpdate]:
		"""Collect parameters from a sampled set of clients."""
		selected = self._sample(self.clients, self.fraction_fit)
		updates = [self.send_model(client, round_number) for client in selected]
		if not updates:
			raise RuntimeError("no client parameters were collected")
		return updates

	def aggregate(self, updates: Sequence[ClientUpdate]) -> list[np.ndarray]:
		"""Apply weighted FedAvg using each client's example count."""
		updates = [update for update in updates if update.num_examples > 0]
		if not updates:
			raise ValueError("updates must contain at least one example")
		parameter_count = len(updates[0].parameters)
		if any(len(update.parameters) != parameter_count for update in updates):
			raise ValueError("client parameter structures do not match")
		total = sum(update.num_examples for update in updates)
		aggregated = [
			sum(
				np.asarray(update.parameters[index]) * update.num_examples
				for update in updates
			) / total
			for index in range(parameter_count)
		]
		self.global_parameters = self._copy(aggregated)
		self.model.set_parameters(self._copy(self.global_parameters))
		return self._copy(self.global_parameters)

	def evaluate_global(self, round_number: int) -> Mapping[str, Any]:
		"""Evaluate the global model after a round."""
		if self.global_parameters is None:
			raise RuntimeError("global model is not initialized")
		parameters = self._copy(self.global_parameters)
		if self.evaluate_fn is not None:
			return dict(self.evaluate_fn(parameters, round_number))

		results = [
			client.evaluate(self._copy(parameters), {"round": round_number})
			for client in self._sample(self.clients, self.fraction_evaluate)
			if hasattr(client, "evaluate")
		]
		if not results:
			return {}
		total = sum(int(result[1]) for result in results)
		return {"loss": sum(float(r[0]) * int(r[1]) for r in results) / total}

	def run(self) -> list[Mapping[str, Any]]:
		"""Execute all federated rounds and return evaluation history."""
		self.initialize_global_model()
		history = []
		for round_number in range(1, self.rounds + 1):
			self.aggregate(self.collect_parameters(round_number))
			history.append({"round": round_number, **self.evaluate_global(round_number)})
		return history

	def _sample(self, clients: Sequence[Any], fraction: float) -> list[Any]:
		count = max(1, int(np.ceil(len(clients) * fraction)))
		return self._random.sample(list(clients), count)

	@staticmethod
	def _copy(parameters: Parameters) -> list[np.ndarray]:
		return [np.array(parameter, copy=True) for parameter in parameters]
