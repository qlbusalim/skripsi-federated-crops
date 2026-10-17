"""Cleaning and normalization for raw NASA POWER weather files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .io import canonical_province, csv_files


MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
PARAMETERS = ("PRECTOTCORR", "RH2M", "T2M")
LOCATION_RE = re.compile(r"Location:\s*Latitude\s+([-+\d.]+)\s+Longitude\s+([-+\d.]+)")
ELEVATION_RE = re.compile(r"Elevation.*?=\s*([-+\d.]+)\s*meters", re.IGNORECASE)


def region_name(path: str | Path) -> str:
    """Convert ``sampang_nasa_power.csv`` to ``Sampang``."""
    stem = Path(path).stem.removesuffix("_nasa_power")
    return " ".join(word.capitalize() for word in stem.split("_"))


def province_name(path: str | Path) -> str:
    """Convert the immediate raw-data folder to a display name."""
    return canonical_province(Path(path).parent.name)


def parse_metadata(path: str | Path) -> dict[str, float]:
    """Read coordinates and elevation from NASA POWER metadata."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    location = next((LOCATION_RE.search(line) for line in lines if line.startswith("Location:")), None)
    elevation = next((ELEVATION_RE.search(line) for line in lines if line.startswith("Elevation")), None)
    if location is None:
        raise ValueError(f"Location metadata is missing: {path}")
    return {
        "latitude": float(location.group(1)),
        "longitude": float(location.group(2)),
        "elevation": float(elevation.group(1)) if elevation else float("nan"),
    }


def parse_nasa_file(path: str | Path) -> pd.DataFrame:
    """Parse one NASA POWER pivot CSV into monthly long-form records."""
    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith("PARAMETER,YEAR,")), None)
    if header_index is None:
        raise ValueError(f"NASA POWER data header is missing: {source}")

    table = pd.read_csv(source, skiprows=header_index)
    expected = {"PARAMETER", "YEAR", *MONTHS}
    if not expected.issubset(table.columns):
        raise ValueError(f"Unexpected NASA POWER columns in {source}")
    table = table[table["PARAMETER"].isin(PARAMETERS)].copy()
    if table.empty:
        raise ValueError(f"No supported parameters found in {source}")

    long = table.melt(
        id_vars=["PARAMETER", "YEAR"],
        value_vars=list(MONTHS),
        var_name="month",
        value_name="value",
    )
    long["date"] = pd.to_datetime(
        long["YEAR"].astype(str) + "-" + long["month"],
        format="%Y-%b",
        errors="coerce",
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce").replace(-999, pd.NA)
    long = long.dropna(subset=["date", "value"])
    metadata = parse_metadata(source)
    long = long.rename(columns={"PARAMETER": "parameter"})
    long["province"] = province_name(source)
    long["region"] = region_name(source)
    for key, value in metadata.items():
        long[key] = value
    return long[["province", "region", "latitude", "longitude", "elevation", "date", "parameter", "value"]]


def load_weather_regions(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load valid files and return ``(records, file_report)``."""
    records: list[pd.DataFrame] = []
    report: list[dict[str, object]] = []
    for path in csv_files(root):
        try:
            frame = parse_nasa_file(path)
            records.append(frame)
            report.append({"file": str(path), "valid": True, "rows": len(frame), "error": ""})
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            report.append({"file": str(path), "valid": False, "rows": 0, "error": str(exc)})
    if not records:
        raise ValueError(f"No valid NASA POWER files found below {root}")
    return pd.concat(records, ignore_index=True), pd.DataFrame(report)


__all__ = ["load_weather_regions", "parse_nasa_file", "parse_metadata", "province_name", "region_name"]
