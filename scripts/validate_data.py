"""Validate raw NASA POWER files and write a validation report."""

from pathlib import Path

from src.data.cleaning import load_weather_regions
from src.data.io import write_table


def main() -> None:
    records, report = load_weather_regions(Path("data/raw"))
    output = write_table(report, "results/validation/raw_data_report.csv")
    print(f"Valid records: {len(records):,}")
    print(f"Files: {len(report)}; valid: {int(report['valid'].sum())}; invalid: {int((~report['valid']).sum())}")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
