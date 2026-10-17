"""Joining prepared weather features with agricultural targets."""

from __future__ import annotations

import pandas as pd


def merge_weather_and_agriculture(
    weather: pd.DataFrame,
    agriculture: pd.DataFrame,
    *,
    keys: tuple[str, ...] = ("province", "date"),
) -> pd.DataFrame:
    """Inner-join weather and agriculture data on aligned period keys."""
    for frame, name in ((weather, "weather"), (agriculture, "agriculture")):
        missing = set(keys).difference(frame.columns)
        if missing:
            raise KeyError(f"{name} is missing columns: {', '.join(sorted(missing))}")
    result = weather.merge(agriculture, on=list(keys), how="inner", suffixes=("", "_agriculture"))
    if result.empty:
        raise ValueError("Weather and agriculture data have no overlapping keys")
    return result.sort_values(list(keys), kind="stable").reset_index(drop=True)


__all__ = ["merge_weather_and_agriculture"]
