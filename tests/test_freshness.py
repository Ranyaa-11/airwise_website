import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from services.freshness import IST, source_age_minutes


class SourceAgeTests(unittest.TestCase):
    def test_naive_timestamp_is_interpreted_as_ist(self):
        now = datetime(2026, 10, 8, 12, 0, tzinfo=IST)
        self.assertEqual(
            source_age_minutes("2026-10-08T10:30:00", now),
            90,
        )

    def test_utc_timestamp_is_converted_to_ist(self):
        now = datetime(2026, 10, 8, 12, 0, tzinfo=IST)
        self.assertEqual(
            source_age_minutes("2026-10-08T05:00:00Z", now),
            90,
        )

    def test_date_only_source_is_ist_midnight_not_current_wall_clock(self):
        now = datetime(2026, 10, 8, 12, 0, tzinfo=IST)
        self.assertEqual(source_age_minutes("2026-10-08", now), 720)

    def test_future_source_age_is_not_negative(self):
        now = datetime(2026, 10, 8, 12, 0, tzinfo=IST)
        self.assertEqual(
            source_age_minutes(datetime(2026, 10, 8, 12, 1, tzinfo=ZoneInfo("UTC")), now),
            0,
        )

    def test_missing_or_invalid_source_timestamp_has_no_age(self):
        self.assertIsNone(source_age_minutes(None))
        self.assertIsNone(source_age_minutes("not a timestamp"))


if __name__ == "__main__":
    unittest.main()
