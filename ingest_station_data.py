"""Append daily station-export drops to the district workbooks.

Expected input: one <City>.csv or <City>.xlsx file for every configured city
inside data/incoming/. Files must contain From Date and all CPCB pollutant
columns. This script imports supplied station exports; it does not download
station data from an external service.
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from services import data_service as ds
from services.aqi_service import POLLUTANTS

DATE_COLUMN = "From Date"
STATION_COLUMNS = [DATE_COLUMN, *POLLUTANTS]
IST = ZoneInfo("Asia/Kolkata")


def load_station_file(path):
    """Load and validate a city station export."""
    if path.lower().endswith(".csv"):
        frame = pd.read_csv(path, encoding="utf-8-sig")
    elif path.lower().endswith(".xlsx"):
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported station file type: {path}")

    missing = [column for column in STATION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    frame = frame[STATION_COLUMNS].copy()
    frame[DATE_COLUMN] = pd.to_datetime(
        frame[DATE_COLUMN], format="mixed", dayfirst=True, errors="coerce"
    ).dt.normalize()
    if frame[DATE_COLUMN].isna().any():
        raise ValueError(f"{path} contains invalid or empty '{DATE_COLUMN}' values")

    for pollutant in POLLUTANTS:
        frame[pollutant] = pd.to_numeric(frame[pollutant], errors="coerce")
    if frame[POLLUTANTS].isna().all(axis=None):
        raise ValueError(f"{path} contains no numeric pollutant observations")

    return frame.drop_duplicates(subset=[DATE_COLUMN], keep="last").sort_values(
        DATE_COLUMN
    ).reset_index(drop=True)


def merge_station_data(existing, incoming):
    """Append new rows, replacing duplicate dates with the incoming record."""
    combined = pd.concat(
        [existing[STATION_COLUMNS], incoming[STATION_COLUMNS]],
        ignore_index=True,
    )
    combined[DATE_COLUMN] = pd.to_datetime(
        combined[DATE_COLUMN], format="mixed", dayfirst=True, errors="coerce"
    ).dt.normalize()
    if combined[DATE_COLUMN].isna().any():
        raise ValueError("Existing station workbook contains invalid dates")
    return combined.drop_duplicates(subset=[DATE_COLUMN], keep="last").sort_values(
        DATE_COLUMN
    ).reset_index(drop=True)


def validate_latest_dates(frames, today, max_age_days=2):
    """Fail with district-level detail if any latest station date is stale."""
    stale = []
    for city in ds.CITIES:
        if city not in frames or frames[city].empty:
            stale.append(f"{city}: no station rows")
            continue
        latest = pd.to_datetime(
            frames[city][DATE_COLUMN], format="mixed", dayfirst=True, errors="coerce"
        ).max().date()
        age_days = (today - latest).days
        if age_days < 0:
            stale.append(f"{city}: latest date {latest} is in the future")
        elif age_days > max_age_days:
            stale.append(
                f"{city}: latest date {latest} is {age_days} days old "
                f"(maximum {max_age_days})"
            )
    if stale:
        raise ValueError("Station data freshness check failed:\n  " + "\n  ".join(stale))


def _input_file(input_dir, city):
    matches = [
        os.path.join(input_dir, f"{city}{extension}")
        for extension in (".csv", ".xlsx")
        if os.path.isfile(os.path.join(input_dir, f"{city}{extension}"))
    ]
    if len(matches) != 1:
        if not matches:
            raise FileNotFoundError(
                f"Missing incoming export for {city}; expected {city}.csv or {city}.xlsx"
            )
        raise ValueError(f"Multiple incoming exports found for {city}: {matches}")
    return matches[0]


def ingest(input_dir, data_dir=ds.DATA_DIR, today=None):
    """Validate all district drops, stage merged workbooks, and replace sources."""
    current_day = today or datetime.now(IST).date()
    incoming = {
        city: load_station_file(_input_file(input_dir, city))
        for city in ds.CITIES
    }
    validate_latest_dates(incoming, current_day)

    merged = {}
    for city in ds.CITIES:
        source = os.path.join(data_dir, f"{city}.xlsx")
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Existing district workbook not found: {source}")
        existing = pd.read_excel(source)
        missing = [column for column in STATION_COLUMNS if column not in existing.columns]
        if missing:
            raise ValueError(
                f"{source} is missing required columns: {', '.join(missing)}"
            )
        merged[city] = merge_station_data(existing, incoming[city])

    validate_latest_dates(merged, current_day)

    staged_paths = {}
    try:
        for city, frame in merged.items():
            target = os.path.join(data_dir, f"{city}.xlsx")
            handle = tempfile.NamedTemporaryFile(
                prefix=f".{city}.", suffix=".xlsx", dir=data_dir, delete=False
            )
            handle.close()
            staged_paths[target] = handle.name
            frame.to_excel(handle.name, index=False)

        for target, staged in staged_paths.items():
            os.replace(staged, target)
    finally:
        for staged in staged_paths.values():
            if os.path.exists(staged):
                os.remove(staged)

    return {
        city: {
            "rows": len(incoming[city]),
            "latest": merged[city][DATE_COLUMN].max().strftime("%Y-%m-%d"),
        }
        for city in ds.CITIES
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=os.path.join(ds.DATA_DIR, "incoming"),
        help="directory containing one fresh CSV/XLSX export per district",
    )
    args = parser.parse_args()
    try:
        result = ingest(args.input_dir)
    except Exception as exc:
        print(f"Station ingestion failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    for city, status in result.items():
        print(f"{city:<16} appended/updated {status['rows']:>4} rows; latest {status['latest']}")


if __name__ == "__main__":
    main()
