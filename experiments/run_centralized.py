"""Run independent centralized training for each simulated client.

Here, centralized means that each client trains locally on its own complete
dataset without model exchange or FedAvg aggregation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.sequence import make_sequences
from src.data.splitter import split_sequences_by_time
from src.evaluation.metrics import calculate_metrics
from src.federated.partition import partition_by_province
from src.models.factory import create_model
from src.preprocessing.scaler import TrainOnlyScaler
from src.training.callbacks import ReduceLROnPlateau
from src.training.loss import get_loss
from src.training.trainer import Trainer, create_optimizer, create_scheduler
from src.utils.config import load_config
from src.utils.seed import set_seed


def _scale_splits(
    splits: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]], TrainOnlyScaler]:
    """Fit one scaler on a client's train split and transform local splits."""
    scaler = TrainOnlyScaler()
    X_train, y_train, _ = splits["train"]
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]), y_train)

    scaled = {}
    for name, (X_part, y_part, metadata) in splits.items():
        X_scaled = scaler.transform_features(
            X_part.reshape(-1, X_part.shape[-1])
        ).reshape(X_part.shape).astype(np.float32)
        y_scaled = scaler.transform_target(y_part).astype(np.float32)
        scaled[name] = (X_scaled, y_scaled, metadata)
    return scaled, scaler


def _prepare_client(client_data: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Create local sequences, temporal splits, and local scaling."""
    data_config = config["data"]
    X, y, metadata = make_sequences(
        client_data,
        feature_columns=data_config["feature_columns"],
        target_column=data_config["target_column"],
        window_size=data_config["sequence"]["window_size"],
        horizon=data_config["sequence"]["horizon"],
        group_column=data_config.get("group_column", "province"),
        date_column=data_config["date_column"],
    )

    split_config = config.get("split", {})
    splits = split_sequences_by_time(
        X, y, metadata,
        time_column=data_config["date_column"],
        train_end=split_config.get("train_end", "2023-12-01"),
        validation_end=split_config.get("validation_end", "2024-12-01"),
    )
    scaled_splits, scaler = _scale_splits(splits)
    return {
        "data": client_data,
        "X": X,
        "y": y,
        "metadata": metadata,
        "splits": splits,
        "scaled_splits": scaled_splits,
        "scaler": scaler,
    }


def prepare_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Load the dataset and prepare one local experiment per client."""
    data_config = config["data"]
    data = pd.read_csv(
        config["paths"]["final_dataset"],
        parse_dates=[data_config["date_column"]],
    )
    province_column = data_config.get("group_column", "province")
    clients = partition_by_province(data, province_column=province_column)

    expected_clients = config.get("centralized", {}).get("expected_clients")
    if expected_clients is not None and len(clients) != int(expected_clients):
        raise ValueError(
            f"Expected {expected_clients} clients, but found {len(clients)} "
            f"unique values in '{province_column}'"
        )

    prepared_clients = {
        client_id: _prepare_client(client_data, config)
        for client_id, client_data in clients.items()
    }
    return {
        "data": data,
        "clients": prepared_clients,
        "client_ids": list(prepared_clients),
        "loss": get_loss(config["training"]["loss"]),
    }


def _loader(
    split: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X, y, _ = split
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


@torch.no_grad()
def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions = [model(inputs.to(device)).cpu().numpy() for inputs, _ in loader]
    return np.concatenate(predictions, axis=0)


def _make_scheduler_and_callbacks(optimizer, training: dict[str, Any]):
    """Build fresh scheduling objects for one client's model."""
    scheduler_config = dict(training.get("scheduler", {}))
    scheduler_name = scheduler_config.pop("name", "reduce_on_plateau")
    if scheduler_name.lower().strip() in {
        "reduce_on_plateau", "plateau", "reducelronplateau",
    }:
        callbacks = [
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=float(scheduler_config.get("factor", 0.5)),
                patience=int(scheduler_config.get("patience", 2)),
                min_delta=float(scheduler_config.get("threshold", training.get("min_delta", 0.0))),
                min_lr=float(scheduler_config.get("min_lr", 1e-6)),
            )
        ]
        return None, callbacks
    return create_scheduler(optimizer, name=scheduler_name, **scheduler_config), []


def _run_client(
    client_id: str,
    model_name: str,
    prepared: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Train and evaluate exactly one independently initialized model."""
    set_seed(seed)
    training = config["training"]
    scaled = prepared["scaled_splits"]
    batch_size = int(training.get("batch_size", 32))
    train_loader = _loader(scaled["train"], batch_size, shuffle=True)
    train_eval_loader = _loader(scaled["train"], batch_size, shuffle=False)
    validation_loader = _loader(scaled["validation"], batch_size, shuffle=False)
    test_loader = _loader(scaled["test"], batch_size, shuffle=False)

    device_name = training.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = create_model(config, input_size=prepared["X"].shape[-1])
    optimizer = create_optimizer(
        model,
        name=training.get("optimizer", "adam"),
        lr=float(training.get("learning_rate", 1e-3)),
    )
    scheduler, callbacks = _make_scheduler_and_callbacks(optimizer, training)

    client_dir = output_dir / client_id
    trainer = Trainer(
        model,
        get_loss(training["loss"]),
        optimizer,
        device=device,
        checkpoint_path=str(client_dir / f"{model_name}_best.pt"),
        patience=training.get("patience", 5),
        min_delta=float(training.get("min_delta", 0.0)),
        scheduler=scheduler,
        callbacks=callbacks,
    )
    history = trainer.fit(
        train_loader,
        validation_loader,
        int(training.get("epochs", 20)),
        verbose=int(training.get("verbose", 1)),
    )

    training_predictions_scaled = _predict(model, train_eval_loader, trainer.device)
    testing_predictions_scaled = _predict(model, test_loader, trainer.device)
    _, y_train_scaled, _ = scaled["train"]
    _, y_test_scaled, metadata_test = scaled["test"]
    training_actual = prepared["scaler"].inverse_transform(y_train_scaled)
    training_predicted = prepared["scaler"].inverse_transform(training_predictions_scaled)
    testing_actual = prepared["scaler"].inverse_transform(y_test_scaled)
    testing_predicted = prepared["scaler"].inverse_transform(testing_predictions_scaled)
    training_metrics = calculate_metrics(training_actual, training_predicted)
    testing_metrics = calculate_metrics(testing_actual, testing_predicted)
    metrics = {
        "training_mae": training_metrics["mae"],
        "training_rmse": training_metrics["rmse"],
        "testing_mae": testing_metrics["mae"],
        "testing_rmse": testing_metrics["rmse"],
    }

    province_column = config["data"].get("group_column", "province")
    province = str(prepared["data"][province_column].iloc[0])
    client_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(client_dir / "history.csv", index=False)
    predictions_frame = metadata_test.copy()
    predictions_frame.insert(0, "model_name", model_name)
    predictions_frame.insert(0, "client_id", client_id)
    predictions_frame["actual"] = testing_actual.reshape(-1)
    predictions_frame["predicted"] = testing_predicted.reshape(-1)
    predictions_frame.to_csv(client_dir / "test_predictions.csv", index=False)
    pd.DataFrame([{
        "model_name": model_name,
        "client_id": client_id,
        "province": province,
        **metrics,
    }]).to_csv(client_dir / "test_metrics.csv", index=False)

    return {
        "model_name": model_name,
        "client_id": client_id,
        "province": province,
        "n_sequences": len(prepared["X"]),
        "n_train": len(scaled["train"][0]),
        "n_validation": len(scaled["validation"][0]),
        "n_test": len(scaled["test"][0]),
        "metrics": metrics,
        "checkpoint": client_dir / f"{model_name}_best.pt",
        "history": history,
    }


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Train and evaluate one independent model for every prepared client."""
    prepared = prepare_experiment(config)
    output_dir = Path(config["paths"].get("output_dir", "results")) / "centralized"
    base_seed = int(config.get("project", {}).get("seed", 42))

    results = []
    for index, client_id in enumerate(prepared["client_ids"]):
        model_name = f"{config['model']['name'].lower()}_{index + 1}"
        result = _run_client(
            client_id,
            model_name,
            prepared["clients"][client_id],
            config,
            output_dir,
            seed=base_seed + index,
        )
        results.append(result)
        print(
            f"{result['model_name']} ({result['province']}) complete - "
            f"train MAE={result['metrics']['training_mae']:.6f}, "
            f"test MAE={result['metrics']['testing_mae']:.6f}"
        )

    metrics_frame = pd.DataFrame([
        {
            "model_name": result["model_name"],
            "client_id": result["client_id"],
            "province": result["province"],
            "n_sequences": result["n_sequences"],
            "n_train": result["n_train"],
            "n_validation": result["n_validation"],
            "n_test": result["n_test"],
            **result["metrics"],
        }
        for result in results
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(output_dir / "all_client_metrics.csv", index=False)
    return {
        "prepared": prepared,
        "results": results,
        "metrics": metrics_frame,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one independent centralized model per client"
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    result = run_experiment(load_config(args.config))
    print("Centralized per-client experiment complete")
    print(f"Clients trained: {len(result['results'])}")
    print(f"Metrics: {result['output_dir'] / 'all_client_metrics.csv'}")


if __name__ == "__main__":
    main()
