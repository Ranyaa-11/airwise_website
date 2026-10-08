import unittest

import numpy as np
import pandas as pd

from services.aqi_service import POLLUTANTS
from train_direct_forecast import (
    build_origin_features,
    chronological_split,
    feature_names,
)


def sample_daily(days=40):
    frame = pd.DataFrame({
        "City": "Bengaluru",
        "From Date": pd.date_range("2026-01-01", periods=days, freq="D"),
        "AQI": np.arange(days, dtype=float) + 20,
    })
    for pollutant in POLLUTANTS:
        frame[pollutant] = np.arange(days, dtype=float) + 10
    return frame


class DirectForecastFeatureTests(unittest.TestCase):
    def test_pollutant_feature_names_are_lagged_or_rolling(self):
        names = feature_names()
        for pollutant in POLLUTANTS:
            self.assertNotIn(pollutant, names)
            self.assertTrue(any(
                name.startswith(f"{pollutant}_LAG_") for name in names
            ))
            self.assertTrue(any(
                name.startswith(f"{pollutant}_ROLL_MEAN_") for name in names
            ))

    def test_current_day_pollutant_change_does_not_change_origin_inputs(self):
        original = sample_daily()
        changed = original.copy()
        origin_date = pd.Timestamp("2026-01-25")
        changed.loc[changed["From Date"] == origin_date, "PM2.5"] = 999999

        original_features = build_origin_features(original)
        changed_features = build_origin_features(changed)
        original_row = original_features[
            (original_features["OriginDate"] == origin_date)
            & (original_features["Horizon"] == 1)
        ].iloc[0]
        changed_row = changed_features[
            (changed_features["OriginDate"] == origin_date)
            & (changed_features["Horizon"] == 1)
        ].iloc[0]

        for name in feature_names():
            if pd.isna(original_row[name]):
                self.assertTrue(pd.isna(changed_row[name]), name)
            else:
                self.assertEqual(original_row[name], changed_row[name], name)

    def test_target_date_calendar_is_known_and_horizon_specific(self):
        samples = build_origin_features(sample_daily())
        origin = pd.Timestamp("2026-01-20")
        one_day = samples[
            (samples["OriginDate"] == origin) & (samples["Horizon"] == 1)
        ].iloc[0]
        two_day = samples[
            (samples["OriginDate"] == origin) & (samples["Horizon"] == 2)
        ].iloc[0]

        self.assertEqual(one_day["TargetDate"], origin + pd.Timedelta(days=1))
        self.assertEqual(two_day["TargetDate"], origin + pd.Timedelta(days=2))
        self.assertNotEqual(
            (one_day["TARGET_DOW_SIN"], one_day["TARGET_DOW_COS"]),
            (two_day["TARGET_DOW_SIN"], two_day["TARGET_DOW_COS"]),
        )

    def test_split_is_chronological_and_targets_do_not_cross_validation_start(self):
        samples = build_origin_features(sample_daily())
        split = chronological_split(samples)
        horizon_one = samples[
            (samples["Horizon"] == 1) & samples["TargetAQI"].notna()
        ]
        train = horizon_one[horizon_one["TargetDate"] < split]
        validation = horizon_one[horizon_one["OriginDate"] >= split]

        self.assertLess(train["TargetDate"].max(), split)
        self.assertGreaterEqual(validation["OriginDate"].min(), split)


if __name__ == "__main__":
    unittest.main()
