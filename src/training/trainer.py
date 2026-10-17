"""Common PyTorch training utilities for centralized and federated learning."""

from __future__ import annotations

import copy
import os
from typing import Any, Callable, Iterable, Mapping, Optional

import torch
from torch import nn


class Trainer:
	"""Train a model, optionally keeping the best validation checkpoint."""

	def __init__(
		self,
		model: nn.Module,
		criterion: Callable[..., torch.Tensor],
		optimizer: torch.optim.Optimizer,
		device: Optional[str | torch.device] = None,
		checkpoint_path: Optional[str] = None,
		patience: Optional[int] = None,
		min_delta: float = 0.0,
		batch_adapter: Optional[Callable[[Any], tuple[Any, Any]]] = None,
	) -> None:
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
		self.model.to(self.device)
		self.checkpoint_path = checkpoint_path
		self.patience = patience
		self.min_delta = min_delta
		self.batch_adapter = batch_adapter or self._unpack_batch
		self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
		self.best_loss = float("inf")

	@staticmethod
	def _unpack_batch(batch: Any) -> tuple[Any, Any]:
		if isinstance(batch, Mapping):
			return batch["inputs"], batch["targets"]
		if isinstance(batch, (tuple, list)) and len(batch) >= 2:
			return batch[0], batch[1]
		raise ValueError("Each batch must be (inputs, targets) or a mapping with inputs/targets")

	def _forward_loss(self, batch: Any) -> torch.Tensor:
		inputs, targets = self.batch_adapter(batch)
		if isinstance(inputs, (tuple, list)):
			inputs = tuple(x.to(self.device) if torch.is_tensor(x) else x for x in inputs)
			output = self.model(*inputs)
		else:
			output = self.model(inputs.to(self.device) if torch.is_tensor(inputs) else inputs)
		if torch.is_tensor(targets):
			targets = targets.to(self.device)
		return self.criterion(output, targets)

	def _batch_size(self, batch: Any) -> int:
		inputs, _ = self.batch_adapter(batch)
		first = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
		return int(first.shape[0]) if torch.is_tensor(first) and first.ndim else 1

	def train_epoch(self, loader: Iterable[Any]) -> float:
		self.model.train()
		total, count = 0.0, 0
		for batch in loader:
			self.optimizer.zero_grad(set_to_none=True)
			loss = self._forward_loss(batch)
			loss.backward()
			self.optimizer.step()
			size = self._batch_size(batch)
			total += loss.detach().item() * size
			count += size
		return total / max(count, 1)

	@torch.no_grad()
	def validate(self, loader: Iterable[Any]) -> float:
		self.model.eval()
		total, count = 0.0, 0
		for batch in loader:
			size = self._batch_size(batch)
			total += self._forward_loss(batch).item() * size
			count += size
		return total / max(count, 1)

	def save_checkpoint(self, epoch: int, path: Optional[str] = None) -> None:
		target = path or self.checkpoint_path
		if not target:
			return
		os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
		torch.save({
			"epoch": epoch,
			"model_state_dict": self.model.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
			"history": self.history,
			"best_loss": self.best_loss,
		}, target)

	def fit(self, train_loader: Iterable[Any], val_loader: Optional[Iterable[Any]], epochs: int) -> dict[str, list[float]]:
		"""Run training with validation, loss history, checkpointing and early stopping."""
		stale = 0
		for epoch in range(epochs):
			train_loss = self.train_epoch(train_loader)
			val_loss = self.validate(val_loader) if val_loader is not None else train_loss
			self.history["train_loss"].append(train_loss)
			self.history["val_loss"].append(val_loss)
			if val_loss < self.best_loss - self.min_delta:
				self.best_loss = val_loss
				self._best_state = copy.deepcopy(self.model.state_dict())
				self.save_checkpoint(epoch)
				stale = 0
			else:
				stale += 1
				if self.patience is not None and stale >= self.patience:
					break
		if hasattr(self, "_best_state"):
			self.model.load_state_dict(self._best_state)
		return self.history

	def train_local(self, loader: Iterable[Any], epochs: int = 1) -> dict[str, torch.Tensor]:
		"""Train a federated client locally and return weights for aggregation."""
		for _ in range(epochs):
			self.train_epoch(loader)
		return copy.deepcopy(self.model.state_dict())


def create_optimizer(model: nn.Module, name: str = "adam", lr: float = 1e-3, **kwargs: Any) -> torch.optim.Optimizer:
	"""Create a supported optimizer for the trainer."""
	choices = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
	if name.lower() not in choices:
		raise ValueError(f"Unsupported optimizer: {name}")
	return choices[name.lower()](model.parameters(), lr=lr, **kwargs)
