"""Normalize regional weather files and aggregate them by province."""

from pathlib import Path

from src.data.aggregation import aggregate_weather_by_province
from src.data.cleaning import load_weather_regions
from src.data.io import write_table


def main() -> None:
    regions, report = load_weather_regions(Path("data/raw"))
    regional_path = write_table(regions, "data/processed/weather_monthly_by_region.csv")
    provincial = aggregate_weather_by_province(regions)
    provincial_path = write_table(provincial, "data/processed/weather_monthly_by_province.csv")
    write_table(report, "results/validation/raw_data_report.csv")
    print(f"Regional weather: {regional_path}")
    print(f"Provincial weather: {provincial_path}")
    print(f"Skipped invalid files: {int((~report['valid']).sum())}")


if __name__ == "__main__":
    main()
