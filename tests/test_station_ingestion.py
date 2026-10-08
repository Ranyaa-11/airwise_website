import unittest
from datetime import date
import tempfile
from pathlib import Path

import pandas as pd

from ingest_station_data import (
    STATION_COLUMNS,
    load_station_file,
    merge_station_data,
    validate_latest_dates,
)
from services.data_service import CITIES


def station_frame(date_value, pm25):
    return pd.DataFrame(
        [[date_value, pm25, 20, 10, 10, 0.5, 20]],
        columns=STATION_COLUMNS,
    )


class StationIngestionTests(unittest.TestCase):
    def test_load_station_csv_keeps_iso_calendar_dates(self):
        frame = station_frame("2026-10-08", 35)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Bengaluru.csv"
            frame.to_csv(path, index=False)

            loaded = load_station_file(str(path))

        self.assertEqual(
            loaded.iloc[0]["From Date"].strftime("%Y-%m-%d"),
            "2026-10-08",
        )

    def test_append_deduplicates_dates_and_incoming_row_wins(self):
        existing = station_frame("2026-10-07", 10)
        incoming = station_frame("2026-10-07", 35)
        incoming = pd.concat(
            [incoming, station_frame("2026-10-08", 40)],
            ignore_index=True,
        )

        merged = merge_station_data(existing, incoming)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.iloc[0]["PM2.5"], 35)
        self.assertEqual(merged.iloc[1]["PM2.5"], 40)

    def test_freshness_rejects_any_district_over_two_days_old(self):
        frames = {
            city: station_frame("2026-10-08", 10)
            for city in CITIES
        }
        frames["Bagalkot"] = station_frame("2026-10-05", 10)

        with self.assertRaises(ValueError) as error:
            validate_latest_dates(frames, date(2026, 10, 8))
        self.assertIn("Bagalkot", str(error.exception))
        self.assertIn("3 days old", str(error.exception))

    def test_freshness_accepts_latest_dates_within_two_days(self):
        frames = {
            city: station_frame("2026-10-06", 10)
            for city in CITIES
        }

        validate_latest_dates(frames, date(2026, 10, 8))


if __name__ == "__main__":
    unittest.main()
