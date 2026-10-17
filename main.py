"""Command-line entry point for the repository framework."""

from __future__ import annotations

import argparse

from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated crops production prediction")
    parser.add_argument(
        "command",
        choices=["prepare-data", "prepare-weather", "prepare-agriculture", "build-dataset", "centralized", "federated"],
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "prepare-weather":
        from scripts.prepare_weather import main as run
        run()
    elif args.command == "prepare-agriculture":
        from scripts.prepare_agriculture import main as run
        run()
    elif args.command == "build-dataset":
        from scripts.build_dataset import main as run
        run()
    elif args.command == "prepare-data":
        from scripts.prepare_weather import main as prepare_weather
        from scripts.prepare_agriculture import main as prepare_agriculture
        from scripts.build_dataset import main as build_dataset
        prepare_weather()
        prepare_agriculture()
        build_dataset()
    elif args.command == "centralized":
        from experiments.run_centralized import prepare_experiment
        prepared = prepare_experiment(config)
        print("Centralized experiment skeleton ready")
        print(f"Model: {config['model']['name']}")
        print(f"Sequences: {len(prepared['X'])}; shape: {prepared['X'].shape}")
        print("Training and temporal split are not activated yet.")
    else:
        print("Federated experiment skeleton ready")
        print(f"Mode: {config['project']['mode']}")
        print(f"Rounds: {config['federated']['rounds']}")
        print("Client construction, dataloaders, and Flower simulation are not activated yet.")


if __name__ == "__main__":
    main()
