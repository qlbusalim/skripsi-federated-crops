"""Federated-learning client skeleton for provincial regression."""

from collections.abc import Iterable

import flwr as fl
import numpy as np
import torch
from torch import nn

from src.evaluation.metrics import mae, rmse


class FederatedClient(fl.client.NumPyClient):
    """Flower client for local MSE training and MAE/RMSE evaluation."""

    def __init__(
        self,
        model: nn.Module,
        trainloader: Iterable,
        testloader: Iterable,
        device: str | torch.device = "cpu",
        epochs: int = 1,
        learning_rate: float = 1e-3,
    ) -> None:
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = torch.device(device)
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def get_parameters(self, config: dict[str, object] | None = None) -> list[np.ndarray]:
        return [value.detach().cpu().numpy().copy() for value in self.model.state_dict().values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        state_dict = self.model.state_dict()
        state_dict.update({
            name: torch.as_tensor(value, device=self.device)
            for name, value in zip(state_dict.keys(), parameters, strict=True)
        })
        self.model.load_state_dict(state_dict)

    def fit(self, parameters: list[np.ndarray], config: dict[str, object]):
        """Train the received global model locally; aggregation is server-side."""
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(int(config.get("epochs", self.epochs))):
            for inputs, targets in self.trainloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).float().reshape(-1, 1)
                self.optimizer.zero_grad(set_to_none=True)
                self.criterion(self.model(inputs), targets).backward()
                self.optimizer.step()
        return self.get_parameters(), self._sample_count(self.trainloader), {}

    def evaluate(self, parameters: list[np.ndarray], config: dict[str, object]):
        """Evaluate the global model using regression metrics."""
        self.set_parameters(parameters)
        self.model.eval()
        actual, predicted = [], []
        total_loss, samples = 0.0, 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).float().reshape(-1, 1)
                outputs = self.model(inputs)
                size = targets.size(0)
                total_loss += self.criterion(outputs, targets).item() * size
                actual.extend(targets.cpu().numpy().reshape(-1))
                predicted.extend(outputs.cpu().numpy().reshape(-1))
                samples += size
        if not samples:
            return 0.0, 0, {"mae": 0.0, "rmse": 0.0}
        return total_loss / samples, samples, {"mae": mae(actual, predicted), "rmse": rmse(actual, predicted)}

    @staticmethod
    def _sample_count(loader: Iterable) -> int:
        dataset = getattr(loader, "dataset", None)
        if dataset is not None:
            return len(dataset)
        return sum(len(targets) for _, targets in loader)
