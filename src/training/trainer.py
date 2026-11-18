"""Common PyTorch training utilities for centralized and federated learning."""

from __future__ import annotations

import copy
import os
import sys
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn

from src.training.callbacks import Callback


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
		scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
		callbacks: Optional[Sequence[Callback]] = None,
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
		self.scheduler = scheduler
		self.callbacks = list(callbacks or [])
		self.history: dict[str, list[float]] = {
			"epoch": [], "train_loss": [], "val_loss": [], "learning_rate": [],
		}
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

	@staticmethod
	def _progress(loader: Iterable[Any], label: str) -> Iterable[Any]:
		"""Yield batches while rendering a dependency-free terminal progress bar."""
		try:
			total = len(loader)  # type: ignore[arg-type]
		except TypeError:
			total = None
		width = 24
		for index, batch in enumerate(loader, start=1):
			if total:
				fraction = min(index / total, 1.0)
				filled = int(width * fraction)
				bar = "=" * filled + ">" + " " * max(0, width - filled - 1)
				text = f"\r{label} [{bar}] {index}/{total}"
			else:
				text = f"\r{label} batches: {index}"
			sys.stdout.write(text)
			sys.stdout.flush()
			yield batch
		sys.stdout.write("\n")
		sys.stdout.flush()

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
			"scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
		}, target)

	def fit(
		self,
		train_loader: Iterable[Any],
		val_loader: Optional[Iterable[Any]],
		epochs: int,
		verbose: int = 0,
	) -> dict[str, list[float]]:
		"""Run training with optional epoch logging.

		``verbose=0`` disables logging and ``verbose=1`` shows batch progress
		bars plus the loss summary for every epoch.
		"""
		if not isinstance(verbose, int) or isinstance(verbose, bool) or verbose < 0:
			raise ValueError("verbose must be a non-negative integer")
		stale = 0
		for epoch in range(epochs):
			if verbose >= 1:
				train_source = self._progress(train_loader, f"Epoch {epoch + 1:03d}/{epochs:03d} train")
				val_source = (
					self._progress(val_loader, f"Epoch {epoch + 1:03d}/{epochs:03d} val")
					if val_loader is not None else None
				)
			else:
				train_source, val_source = train_loader, val_loader
			train_loss = self.train_epoch(train_source)
			val_loss = self.validate(val_source) if val_source is not None else train_loss
			self.history["epoch"].append(epoch + 1)
			self.history["train_loss"].append(train_loss)
			self.history["val_loss"].append(val_loss)
			if self.scheduler is not None:
				if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
					self.scheduler.step(val_loss)
				else:
					self.scheduler.step()
			logs = {"train_loss": train_loss, "val_loss": val_loss}
			for callback in self.callbacks:
				callback.on_epoch_end(epoch, logs=logs, model=self.model, optimizer=self.optimizer)
			current_lr = self.optimizer.param_groups[0]["lr"]
			self.history["learning_rate"].append(current_lr)
			is_best = val_loss < self.best_loss - self.min_delta
			if verbose >= 1:
				status = " *best*" if is_best else ""
				print(
					f"Epoch {epoch + 1:03d}/{epochs:03d} - "
					f"train_loss={train_loss:.6f} - val_loss={val_loss:.6f} - "
					f"lr={current_lr:.6g}{status}"
				)
			if is_best:
				self.best_loss = val_loss
				self._best_state = copy.deepcopy(self.model.state_dict())
				self.save_checkpoint(epoch)
				stale = 0
			else:
				stale += 1
				if self.patience is not None and stale >= self.patience:
					break
			if any(getattr(callback, "stop_training", False) for callback in self.callbacks):
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


def create_scheduler(
	optimizer: torch.optim.Optimizer,
	name: str = "reduce_on_plateau",
	**kwargs: Any,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
	"""Create a configurable learning-rate scheduler.

	The default is validation-aware ``ReduceLROnPlateau``. ``step`` and
	``cosine`` are also available for later experiments; ``none`` disables it.
	"""
	key = name.lower().strip()
	if key in {"none", "off", "disabled"}:
		return None
	if key in {"reduce_on_plateau", "plateau", "reducelronplateau"}:
		return torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer,
			mode=kwargs.get("mode", "min"),
			factor=float(kwargs.get("factor", 0.5)),
			patience=int(kwargs.get("patience", 2)),
			threshold=float(kwargs.get("threshold", 1e-4)),
			min_lr=float(kwargs.get("min_lr", 1e-6)),
		)
	if key == "step":
		return torch.optim.lr_scheduler.StepLR(
			optimizer,
			step_size=int(kwargs.get("step_size", 10)),
			gamma=float(kwargs.get("gamma", 0.1)),
		)
	if key == "cosine":
		return torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=int(kwargs.get("t_max", 10)),
			eta_min=float(kwargs.get("eta_min", 0.0)),
		)
	raise ValueError("Unsupported scheduler: choose reduce_on_plateau, step, cosine, or none")
