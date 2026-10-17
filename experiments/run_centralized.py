"""Centralized experiment skeleton for comparing recurrent models."""

from __future__ import annotations

import argparse

import pandas as pd

from src.data.sequence import make_sequences
from src.models.factory import create_model
from src.training.loss import get_loss
from src.utils.config import load_config


def prepare_experiment(config: dict) -> dict:
    """Load the final data and instantiate the configured model.

    Training and temporal splitting are intentionally left as the next stage.
    """
    data = pd.read_csv(config["paths"]["final_dataset"], parse_dates=[config["data"]["date_column"]])
    data_config = config["data"]
    features = data_config["feature_columns"]
    X, y, metadata = make_sequences(
        data,
        feature_columns=features,
        target_column=data_config["target_column"],
        window_size=data_config["sequence"]["window_size"],
        horizon=data_config["sequence"]["horizon"],
        group_column=data_config.get("group_column", "province"),
        date_column=data_config["date_column"],
    )
    model = create_model(config, input_size=X.shape[-1])
    loss = get_loss(config["training"]["loss"])
    return {"data": data, "X": X, "y": y, "metadata": metadata, "model": model, "loss": loss}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare centralized experiment skeleton")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    prepared = prepare_experiment(config)
    print("Centralized experiment skeleton ready")
    print(f"Model: {config['model']['name']}")
    print(f"Sequences: {len(prepared['X'])}; shape: {prepared['X'].shape}")
    print("Training and temporal split are not activated yet.")


if __name__ == "__main__":
    main()
