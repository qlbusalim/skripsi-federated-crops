"""Normalize and validate the provincial agriculture workbook."""

from src.data.agriculture import load_agriculture_workbook, validate_agriculture_data
from src.data.io import write_table


def main() -> None:
    data = load_agriculture_workbook("data/raw/data_paddy.xlsx")
    report = validate_agriculture_data(data)
    if not bool(report["valid"].all()):
        raise ValueError("Agriculture validation failed; inspect results/validation/agriculture_data_report.csv")
    output = write_table(data, "data/processed/agriculture_monthly_by_province.csv")
    report_output = write_table(report, "results/validation/agriculture_data_report.csv")
    print(f"Agriculture data: {output}")
    print(f"Validation report: {report_output}")
    print(f"Rows: {len(data):,}; provinces: {data['province'].nunique()}")


if __name__ == "__main__":
    main()
