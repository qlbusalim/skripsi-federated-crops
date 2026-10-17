"""Federated experiment skeleton."""

from __future__ import annotations

import argparse

from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare federated experiment skeleton")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    print("Federated experiment skeleton ready")
    print(f"Mode: {config['project']['mode']}")
    print(f"Rounds: {config['federated']['rounds']}")
    print("Client construction, dataloaders, and Flower simulation are not activated yet.")


if __name__ == "__main__":
    main()
