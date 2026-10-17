"""Aggregation of cleaned regional data to the province level."""

from __future__ import annotations

import pandas as pd


def aggregate_weather_by_province(data: pd.DataFrame) -> pd.DataFrame:
    """Average weather parameters across regions for each province and month."""
    required = {"province", "date", "parameter", "value"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing columns: {', '.join(sorted(missing))}")
    grouped = (
        data.groupby(["province", "date", "parameter"], as_index=False)
        .agg(value=("value", "mean"), n_regions=("region", "nunique"))
    )
    result = grouped.pivot_table(
        index=["province", "date"], columns="parameter", values="value", aggfunc="first"
    ).reset_index()
    counts = grouped.groupby(["province", "date"], as_index=False)["n_regions"].max()
    result = result.merge(counts, on=["province", "date"], how="left")
    result.columns.name = None
    return result.sort_values(["province", "date"], kind="stable").reset_index(drop=True)


__all__ = ["aggregate_weather_by_province"]
