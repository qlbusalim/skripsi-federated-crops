"""Loading, cleaning, and validating provincial agriculture data."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

from .io import canonical_province


NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "package_rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
PROVINCE_SHEETS = ("Jawa Barat", "Jawa Tengah", "DI Yogyakarta", "Jawa Timur", "Banten")
EXPECTED_COLUMNS = ("date", "province", "harvested_area", "paddy_prod")


def _column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference.upper())
    if letters is None:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference}")
    index = 0
    for letter in letters.group(0):
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(text.text or "" for text in item.iter(f"{{{NS['main']}}}t")) for item in root.findall("main:si", NS)]


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    value = cell.find("main:v", NS)
    if value is None:
        inline = cell.find("main:is", NS)
        return "" if inline is None else "".join(text.text or "" for text in inline.iter(f"{{{NS['main']}}}t"))
    raw = value.text or ""
    if cell.attrib.get("t") == "s":
        return shared[int(raw)]
    return raw


def _sheet_rows(archive: zipfile.ZipFile, sheet_path: str, shared: list[str]) -> list[list[str]]:
    root = ET.fromstring(archive.read(sheet_path))
    rows: list[list[str]] = []
    for row in root.findall(".//main:sheetData/main:row", NS):
        cells = row.findall("main:c", NS)
        if not cells:
            continue
        values = [""] * (max(_column_index(cell.attrib["r"]) for cell in cells) + 1)
        for cell in cells:
            values[_column_index(cell.attrib["r"])] = _cell_value(cell, shared)
        rows.append(values)
    return rows


def _workbook_sheets(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relation_map = {
        relation.attrib["Id"]: relation.attrib["Target"]
        for relation in relationships.findall("package_rel:Relationship", NS)
    }
    result = {}
    for sheet in workbook.findall("main:sheets/main:sheet", NS):
        relation_id = sheet.attrib[f"{{{NS['rel']}}}id"]
        target = relation_map[relation_id].lstrip("/")
        result[sheet.attrib["name"]] = target if target.startswith("xl/") else f"xl/{target}"
    return result


def _excel_date(value: str) -> pd.Timestamp:
    serial = float(value)
    return pd.Timestamp(datetime(1899, 12, 30) + timedelta(days=serial))


def _parse_number(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().replace(" ", "")
    if not text:
        return float("nan")
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    return float(text)


def load_agriculture_workbook(
    path: str | Path,
    sheets: tuple[str, ...] = PROVINCE_SHEETS,
) -> pd.DataFrame:
    """Load selected provincial sheets from the workbook into one table."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Agriculture workbook not found: {source}")

    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(source) as archive:
        shared = _shared_strings(archive)
        available = _workbook_sheets(archive)
        missing = set(sheets).difference(available)
        if missing:
            raise ValueError(f"Missing province sheet(s): {', '.join(sorted(missing))}")
        for province in sheets:
            rows = _sheet_rows(archive, available[province], shared)
            source_columns = ("date", "region", "harvested_area", "paddy_prod")
            if not rows or tuple(rows[0][:4]) != source_columns:
                raise ValueError(f"Unexpected columns in sheet: {province}")
            frame = pd.DataFrame(rows[1:], columns=rows[0])
            frame = frame.iloc[:, :4].copy()
            frame = frame.rename(columns={"region": "province"})
            frame["province"] = canonical_province(province)
            frame["date"] = frame["date"].map(_excel_date)
            frame["harvested_area"] = frame["harvested_area"].map(_parse_number)
            frame["paddy_prod"] = frame["paddy_prod"].map(_parse_number)
            frames.append(frame[["date", "province", "harvested_area", "paddy_prod"]])

    return pd.concat(frames, ignore_index=True).sort_values(["province", "date"], kind="stable").reset_index(drop=True)


def validate_agriculture_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return one validation row per province without modifying the input."""
    required = set(EXPECTED_COLUMNS)
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing agriculture columns: {', '.join(sorted(missing))}")
    rows = []
    for province, group in data.groupby("province", sort=True):
        duplicate_keys = int(group.duplicated(["province", "date"]).sum())
        rows.append({
            "province": province,
            "rows": len(group),
            "date_min": group["date"].min(),
            "date_max": group["date"].max(),
            "missing_values": int(group[list(EXPECTED_COLUMNS)].isna().sum().sum()),
            "negative_values": int((group[["harvested_area", "paddy_prod"]] < 0).sum().sum()),
            "duplicate_periods": duplicate_keys,
            "valid": bool(
                len(group) == 84
                and group[list(EXPECTED_COLUMNS)].notna().all().all()
                and (group[["harvested_area", "paddy_prod"]] >= 0).all().all()
                and duplicate_keys == 0
            ),
        })
    return pd.DataFrame(rows)


__all__ = ["PROVINCE_SHEETS", "load_agriculture_workbook", "validate_agriculture_data"]
