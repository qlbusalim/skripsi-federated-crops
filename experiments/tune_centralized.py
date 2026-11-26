"""Optuna tuning for the shared centralized model configuration.

Each trial trains one independent model for every client over several repeated
training runs.  The temporal train/validation split stays fixed; only the
training seed changes between repetitions.  The objective is the mean
validation MAE across clients and repetitions.  The test split is never used
while Optuna is searching.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from optuna import Trial, create_study
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader

from experiments.run_centralized import (
    _loader,
    _make_scheduler_and_callbacks,
    _predict,
    prepare_experiment,
)
from src.evaluation.metrics import calculate_metrics
from src.models.factory import create_model
from src.training.loss import get_loss
from src.training.trainer import Trainer, create_optimizer
from src.utils.config import load_config
from src.utils.experiment_paths import centralized_tuning_dir
from src.utils.seed import set_seed


def _suggest_config(trial: Trial, base_config: dict[str, Any]) -> dict[str, Any]:
    """Create a trial configuration from the configured search space."""
    config = copy.deepcopy(base_config)
    tuning = base_config.get("tuning", {})
    search = tuning.get("search_space", {})

    def choices(name: str, fallback: list[Any]) -> list[Any]:
        values = search.get(name, fallback)
        if not isinstance(values, list) or not values:
            raise ValueError(f"tuning.search_space.{name} must be a non-empty list")
        return values

    config["model"]["name"] = trial.suggest_categorical(
        "model_name", choices("model_name", ["rnn", "lstm", "gru", "bilstm", "bigru"])
    )
    config["model"]["hidden_size"] = trial.suggest_categorical(
        "hidden_size", choices("hidden_size", [8, 16, 32])
    )
    config["model"]["num_layers"] = trial.suggest_categorical(
        "num_layers", choices("num_layers", [1, 2])
    )
    config["data"]["sequence"]["window_size"] = trial.suggest_categorical(
        "window_size", choices("window_size", [6, 8, 10, 12])
    )
    config["training"]["optimizer"] = trial.suggest_categorical(
        "optimizer", choices("optimizer", ["adam", "adamw"])
    )
    config["training"]["learning_rate"] = trial.suggest_float(
        "learning_rate",
        float(tuning.get("learning_rate_min", 1e-4)),
        float(tuning.get("learning_rate_max", 1e-2)),
        log=True,
    )
    config["training"]["batch_size"] = trial.suggest_categorical(
        "batch_size", choices("batch_size", [2, 4, 8])
    )
    config["training"]["loss"] = trial.suggest_categorical(
        "loss", choices("loss", ["mae", "mse"])
    )
    config["training"]["verbose"] = 0
    config["training"]["epochs"] = int(tuning.get("epochs", 40))
    config["training"]["device"] = tuning.get(
        "device", config["training"].get("device", "auto")
    )
    return config


def _evaluate_validation(
    model: torch.nn.Module,
    prepared: dict[str, Any],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate validation predictions on the original target scale."""
    predictions_scaled = _predict(model, loader, device)
    _, targets_scaled, _ = prepared["scaled_splits"]["validation"]
    actual = prepared["scaler"].inverse_transform(targets_scaled)
    predicted = prepared["scaler"].inverse_transform(predictions_scaled)
    return calculate_metrics(actual, predicted)


def _train_trial_client(
    client_index: int,
    client_id: str,
    prepared: dict[str, Any],
    config: dict[str, Any],
    trial: Trial,
    repetition_index: int,
) -> dict[str, float]:
    """Train one client for one repeated run without writing result files."""
    base_seed = int(config["project"].get("seed", 42))
    set_seed(base_seed + repetition_index * 100 + client_index)
    training = config["training"]
    batch_size = int(training.get("batch_size", 32))
    scaled = prepared["scaled_splits"]
    train_loader = _loader(scaled["train"], batch_size, shuffle=True)
    validation_loader = _loader(scaled["validation"], batch_size, shuffle=False)

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
    trainer = Trainer(
        model,
        get_loss(training["loss"]),
        optimizer,
        device=device,
        checkpoint_path=None,
        patience=training.get("patience", 5),
        min_delta=float(training.get("min_delta", 0.0)),
        scheduler=scheduler,
        callbacks=callbacks,
    )
    trainer.fit(
        train_loader,
        validation_loader,
        int(training.get("epochs", 40)),
        verbose=0,
    )
    metrics = _evaluate_validation(model, prepared, validation_loader, trainer.device)
    trial.set_user_attr(
        f"repeat_{repetition_index}_{client_id}_validation_mae", metrics["mae"]
    )
    trial.set_user_attr(
        f"repeat_{repetition_index}_{client_id}_validation_rmse", metrics["rmse"]
    )
    return metrics


def make_objective(base_config: dict[str, Any]):
    """Build the Optuna objective over all configured clients."""
    def objective(trial: Trial) -> float:
        trial_config = _suggest_config(trial, base_config)
        prepared = prepare_experiment(trial_config)
        repetitions = int(trial_config.get("tuning", {}).get("cv_repeats", 1))
        if repetitions < 1:
            raise ValueError("tuning.cv_repeats must be at least 1")

        repetition_mae = []
        repetition_rmse = []
        for repetition_index in range(repetitions):
            client_metrics = []
            for index, client_id in enumerate(prepared["client_ids"]):
                metrics = _train_trial_client(
                    index,
                    client_id,
                    prepared["clients"][client_id],
                    trial_config,
                    trial,
                    repetition_index,
                )
                client_metrics.append(metrics)

            current_mae = float(np.mean([item["mae"] for item in client_metrics]))
            current_rmse = float(np.mean([item["rmse"] for item in client_metrics]))
            repetition_mae.append(current_mae)
            repetition_rmse.append(current_rmse)
            trial.report(
                current_mae,
                step=repetition_index,
            )
            if trial.should_prune():
                raise TrialPruned()

        mean_mae = float(np.mean(repetition_mae))
        mean_rmse = float(np.mean(repetition_rmse))
        trial.set_user_attr("mean_validation_mae", mean_mae)
        trial.set_user_attr("mean_validation_rmse", mean_rmse)
        trial.set_user_attr("std_validation_mae", float(np.std(repetition_mae)))
        trial.set_user_attr("std_validation_rmse", float(np.std(repetition_rmse)))
        trial.set_user_attr("validation_mae_by_repeat", repetition_mae)
        trial.set_user_attr("validation_rmse_by_repeat", repetition_rmse)
        return mean_mae

    return objective


def _write_best_config(config: dict[str, Any], best_params: dict[str, Any], path: Path) -> None:
    """Write the best model/training settings as a reusable config file."""
    best_config = copy.deepcopy(config)
    best_config["model"]["name"] = best_params["model_name"]
    best_config["model"]["hidden_size"] = best_params["hidden_size"]
    best_config["model"]["num_layers"] = best_params["num_layers"]
    best_config["data"]["sequence"]["window_size"] = best_params["window_size"]
    best_config["training"]["optimizer"] = best_params["optimizer"]
    best_config["training"]["learning_rate"] = best_params["learning_rate"]
    best_config["training"]["batch_size"] = best_params["batch_size"]
    best_config["training"]["loss"] = best_params["loss"]
    best_config.pop("tuning", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best_config, handle, sort_keys=False)


def run_tuning(config: dict[str, Any]) -> dict[str, Any]:
    """Run the centralized Optuna study and save reusable tuning artifacts."""
    tuning = config.get("tuning", {})
    output_dir = centralized_tuning_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = tuning.get(
        "storage", f"sqlite:///{(output_dir / 'study.sqlite').as_posix()}"
    )
    study = create_study(
        study_name=tuning.get("study_name", "centralized_model_tuning_repeated"),
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        pruner=MedianPruner(
            n_startup_trials=int(tuning.get("pruner_startup_trials", 5)),
            n_warmup_steps=int(tuning.get("pruner_warmup_steps", 2)),
        ),
    )
    study.optimize(
        make_objective(config),
        n_trials=int(tuning.get("n_trials", 30)),
        timeout=tuning.get("timeout_seconds"),
        gc_after_trial=True,
    )

    study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False)
    _write_best_config(config, study.best_params, output_dir / "best_config.yaml")
    summary = {
        "study_name": study.study_name,
        "objective": "mean_validation_mae",
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "best_trial_user_attributes": study.best_trial.user_attrs,
        "completed_trials": sum(
            trial.state.name == "COMPLETE" for trial in study.trials
        ),
    }
    with (output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    return {"study": study, "output_dir": output_dir}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune one shared model configuration across centralized clients"
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    result = run_tuning(load_config(args.config))
    study = result["study"]
    print("Centralized Optuna tuning complete")
    print(f"Best validation MAE: {study.best_value:.6f}")
    print(f"Best parameters: {study.best_params}")
    print(f"Artifacts: {result['output_dir']}")


if __name__ == "__main__":
    main()
