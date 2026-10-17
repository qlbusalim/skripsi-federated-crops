"""Training callbacks used by the federated PyTorch models."""

from pathlib import Path
from typing import Dict, List, Optional

import torch


class Callback:
	"""Base callback invoked at the end of each training epoch."""

	def on_epoch_end(self, epoch: int, logs=None, model=None, optimizer=None) -> None:
		pass


class EarlyStopping(Callback):
	def __init__(self, monitor: str, patience: int, min_delta: float) -> None:
		self.monitor = monitor
		self.patience = patience
		self.min_delta = min_delta
		self.best: Optional[float] = None
		self.wait = 0
		self.stop_training = False

	def on_epoch_end(self, epoch: int, logs=None, model=None, optimizer=None) -> None:
		value = None if logs is None else logs.get(self.monitor)
		if value is None:
			return
		if self.best is None or value < self.best - self.min_delta:
			self.best, self.wait = value, 0
		else:
			self.wait += 1
			if self.wait >= self.patience:
				self.stop_training = True


class ModelCheckpoint(Callback):
	def __init__(self, filepath: Path, monitor: str) -> None:
		self.filepath = filepath
		self.monitor = monitor
		self.best: Optional[float] = None

	def on_epoch_end(self, epoch: int, logs=None, model=None, optimizer=None) -> None:
		if model is None or logs is None or self.monitor not in logs:
			return
		value = logs[self.monitor]
		if self.best is None or value < self.best:
			self.best = value
			torch.save(model.state_dict(), self.filepath)


class ReduceLROnPlateau(Callback):
	def __init__(
		self, monitor: str, factor: float, patience: int, min_delta: float, min_lr: float
	) -> None:
		self.monitor = monitor
		self.factor = factor
		self.patience = patience
		self.min_delta = min_delta
		self.min_lr = min_lr
		self.best: Optional[float] = None
		self.wait = 0

	def on_epoch_end(self, epoch: int, logs=None, model=None, optimizer=None) -> None:
		value = None if logs is None else logs.get(self.monitor)
		if value is None or optimizer is None:
			return
		if self.best is None or value < self.best - self.min_delta:
			self.best, self.wait = value, 0
			return
		self.wait += 1
		if self.wait >= self.patience:
			for group in optimizer.param_groups:
				group["lr"] = max(group["lr"] * self.factor, self.min_lr)
			self.wait = 0


def build_callbacks(
	output_dir: str,
	monitor: str = "val_loss",
	patience: int = 10,
	min_delta: float = 0.0,
	use_lr_scheduler: bool = True,
	lr_patience: int = 4,
) -> List[Callback]:
	"""Create the standard callbacks for a client/server training run.

	The best model is selected using ``monitor``.  A plateau-based learning
	rate scheduler is optional because it may not be useful for every client.
	"""
	checkpoint_dir = Path(output_dir)
	checkpoint_dir.mkdir(parents=True, exist_ok=True)

	callbacks: List[Callback] = [
		EarlyStopping(
			monitor=monitor,
			patience=patience,
			min_delta=min_delta,
		),
		ModelCheckpoint(
			filepath=checkpoint_dir / "best_model.pt",
			monitor=monitor,
		),
	]

	if use_lr_scheduler:
		callbacks.append(
			ReduceLROnPlateau(
				monitor=monitor,
				factor=0.5,
				patience=lr_patience,
				min_delta=min_delta,
				min_lr=1e-6,
			)
		)

	return callbacks
