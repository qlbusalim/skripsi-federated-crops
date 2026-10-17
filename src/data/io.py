"""Input/output helpers used by the data preparation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROVINCE_NAMES = {
    "banten": "Banten",
    "di yogyakarta": "DI Yogyakarta",
    "jawa barat": "Jawa Barat",
    "jawa tengah": "Jawa Tengah",
    "jawa timur": "Jawa Timur",
}


def canonical_province(value: str) -> str:
    """Return the project's canonical spelling for a province name."""
    normalized = " ".join(str(value).replace("_", " ").split()).lower()
    return PROVINCE_NAMES.get(normalized, " ".join(str(value).replace("_", " ").split()))


def csv_files(root: str | Path) -> list[Path]:
    """Return CSV files below *root* in deterministic order."""
    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")
    return sorted(path.rglob("*.csv"))


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a tabular CSV and fail with a useful path in the error."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CSV file not found: {source}")
    try:
        return pd.read_csv(source, **kwargs)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file: {source}") from exc


def write_table(data: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame, creating its parent directory when necessary."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(target, index=False)
    return target


__all__ = ["canonical_province", "csv_files", "read_table", "write_table"]
