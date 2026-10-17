"""Build the final dataset after agriculture data has been prepared."""

from pathlib import Path

from src.data.io import read_table, write_table
from src.data.merging import merge_weather_and_agriculture


def main() -> None:
    weather = read_table("data/processed/weather_monthly_by_province.csv", parse_dates=["date"])
    agriculture_path = Path("data/processed/agriculture_monthly_by_province.csv")
    if not agriculture_path.is_file():
        raise FileNotFoundError(
            f"Agriculture dataset is required before building the final dataset: {agriculture_path}"
        )
    agriculture = read_table(agriculture_path, parse_dates=["date"])
    dataset = merge_weather_and_agriculture(weather, agriculture)
    output = write_table(dataset, "data/processed/final_province_dataset.csv")
    print(f"Final dataset: {output}")


if __name__ == "__main__":
    main()
