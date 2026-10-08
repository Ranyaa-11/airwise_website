"""Train leakage-safe direct AQI models for horizons t+1 through t+7.

Run with ``python train_direct_forecast.py``. Each horizon gets an independent
XGBoost model. Pollutant features only use observations through t-1; AQI history
may include AQI(t), which is known at forecast origin t. The target-date
calendar is included because it is known when the forecast is issued.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from services import aqi_service as aqi
from services.data_service import BASE_DIR, CITIES, DATA_DIR

HORIZONS = range(1, 8)
VALIDATION_FRACTION = 0.2
MODEL_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODEL_DIR, "direct_aqi_metrics.json")
DATE_COLUMN = "From Date"


def load_observed_daily(data_dir=DATA_DIR):
    """Read station observations without the app's full-history imputation."""
    frames = []
    for city in CITIES:
        path = os.path.join(data_dir, f"{city}.xlsx")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing station history workbook: {path}")

        frame = pd.read_excel(path)
        required = {DATE_COLUMN, *aqi.POLLUTANTS}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(sorted(missing))}"
            )

        frame = frame[[DATE_COLUMN, *aqi.POLLUTANTS]].copy()
        frame[DATE_COLUMN] = pd.to_datetime(
            frame[DATE_COLUMN], format="mixed", dayfirst=True, errors="coerce"
        ).dt.normalize()
        frame = frame.dropna(subset=[DATE_COLUMN])
        for pollutant in aqi.POLLUTANTS:
            frame[pollutant] = pd.to_numeric(frame[pollutant], errors="coerce")
            frame.loc[frame[pollutant] < 0, pollutant] = np.nan
        frame["City"] = city
        frames.append(frame)

    observations = pd.concat(frames, ignore_index=True)
    if observations["CO"].notna().any() and observations["CO"].median() > 100:
        observations["CO"] /= 1000.0

    daily = observations.groupby(
        ["City", DATE_COLUMN], as_index=False
    )[aqi.POLLUTANTS].mean()

    def calculate_daily_aqi(row):
        return aqi.calculate_aqi(
            {pollutant: row[pollutant] for pollutant in aqi.POLLUTANTS}
        )["aqi"]

    daily["AQI"] = daily.apply(calculate_daily_aqi, axis=1)
    return daily.sort_values(["City", DATE_COLUMN]).reset_index(drop=True)


def feature_names():
    names = [
        "AQI_LAG_0",
        "AQI_LAG_1",
        "AQI_LAG_2",
        "AQI_LAG_3",
        "AQI_LAG_7",
        "AQI_LAG_14",
        "AQI_ROLL_MEAN_3D",
        "AQI_ROLL_MEAN_7D",
        "AQI_ROLL_MEAN_14D",
        "TARGET_MONTH_SIN",
        "TARGET_MONTH_COS",
        "TARGET_DOW_SIN",
        "TARGET_DOW_COS",
    ]
    for pollutant in aqi.POLLUTANTS:
        names.extend(
            f"{pollutant}_LAG_{lag}" for lag in (1, 2, 3, 7, 14)
        )
        names.extend(
            (f"{pollutant}_ROLL_MEAN_7D", f"{pollutant}_ROLL_MEAN_14D")
        )
    names.extend(f"CITY_{city}" for city in CITIES)
    return names


def build_origin_features(daily):
    """Build features known at t; no pollutant feature includes date t."""
    expected = {"City", DATE_COLUMN, "AQI", *aqi.POLLUTANTS}
    missing = expected.difference(daily.columns)
    if missing:
        raise ValueError(f"Daily data missing columns: {', '.join(sorted(missing))}")

    rows = []
    for city, group in daily.groupby("City", sort=False):
        group = group.sort_values(DATE_COLUMN).drop_duplicates(
            subset=[DATE_COLUMN], keep="last"
        )
        if group.empty:
            continue

        indexed = group.set_index(DATE_COLUMN)
        date_index = pd.date_range(indexed.index.min(), indexed.index.max(), freq="D")
        indexed = indexed.reindex(date_index)
        indexed["City"] = city
        indexed["AQI_ROLL_MEAN_3D"] = indexed["AQI"].rolling(3, min_periods=1).mean()
        indexed["AQI_ROLL_MEAN_7D"] = indexed["AQI"].rolling(7, min_periods=1).mean()
        indexed["AQI_ROLL_MEAN_14D"] = indexed["AQI"].rolling(14, min_periods=1).mean()

        for pollutant in aqi.POLLUTANTS:
            prior = indexed[pollutant].shift(1)
            for lag in (1, 2, 3, 7, 14):
                indexed[f"{pollutant}_LAG_{lag}"] = indexed[pollutant].shift(lag)
            indexed[f"{pollutant}_ROLL_MEAN_7D"] = prior.rolling(
                7, min_periods=1
            ).mean()
            indexed[f"{pollutant}_ROLL_MEAN_14D"] = prior.rolling(
                14, min_periods=1
            ).mean()

        for lag in (0, 1, 2, 3, 7, 14):
            indexed[f"AQI_LAG_{lag}"] = indexed["AQI"].shift(lag)

        origin_rows = indexed.reset_index(names="OriginDate")
        origin_rows = origin_rows[origin_rows["AQI"].notna()]
        for source in origin_rows.to_dict(orient="records"):
            origin_date = pd.Timestamp(source["OriginDate"])
            for horizon in HORIZONS:
                target_date = origin_date + pd.Timedelta(days=horizon)
                target_month_angle = 2 * np.pi * target_date.month / 12
                target_dow_angle = 2 * np.pi * target_date.dayofweek / 7
                features = {
                    name: source.get(name, np.nan) for name in feature_names()
                }
                features.update({
                    "TARGET_MONTH_SIN": np.sin(target_month_angle),
                    "TARGET_MONTH_COS": np.cos(target_month_angle),
                    "TARGET_DOW_SIN": np.sin(target_dow_angle),
                    "TARGET_DOW_COS": np.cos(target_dow_angle),
                    "OriginDate": origin_date,
                    "TargetDate": target_date,
                    "City": city,
                    "Horizon": horizon,
                    "TargetAQI": indexed["AQI"].get(target_date, np.nan),
                    "PersistenceAQI": source["AQI"],
                })
                for city_name in CITIES:
                    features[f"CITY_{city_name}"] = float(city_name == city)
                rows.append(features)

    return pd.DataFrame(rows)


def chronological_split(frame, validation_fraction=VALIDATION_FRACTION):
    origin_dates = np.array(sorted(pd.to_datetime(frame["OriginDate"].unique())))
    if len(origin_dates) < 5:
        raise ValueError("Need at least five distinct forecast-origin dates to split data")
    split_index = int(np.floor(len(origin_dates) * (1 - validation_fraction)))
    split_index = min(max(split_index, 1), len(origin_dates) - 1)
    return pd.Timestamp(origin_dates[split_index])


def train_models(daily, model_dir=MODEL_DIR, metrics_path=METRICS_PATH):
    """Fit and save one direct model per horizon with persistence comparison."""
    if not 0 < VALIDATION_FRACTION < 1:
        raise ValueError("VALIDATION_FRACTION must be between zero and one")

    dataset = build_origin_features(daily)
    split_date = chronological_split(dataset)
    features = feature_names()
    os.makedirs(model_dir, exist_ok=True)
    horizon_metrics = []

    for horizon in HORIZONS:
        samples = dataset[
            (dataset["Horizon"] == horizon) & dataset["TargetAQI"].notna()
        ].copy()
        train = samples[samples["TargetDate"] < split_date]
        validation = samples[samples["OriginDate"] >= split_date]
        if train.empty or validation.empty:
            raise ValueError(
                f"Horizon {horizon}: empty time-based train or validation split "
                f"at {split_date.date()}"
            )

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train[features], train["TargetAQI"])
        model_path = os.path.join(model_dir, f"direct_aqi_horizon_{horizon}.json")
        model.save_model(model_path)

        predicted = model.predict(validation[features])
        targets = validation["TargetAQI"].to_numpy(dtype=float)
        persistence = validation["PersistenceAQI"].to_numpy(dtype=float)
        model_mae = float(np.mean(np.abs(predicted - targets)))
        persistence_mae = float(np.mean(np.abs(persistence - targets)))
        horizon_metrics.append({
            "horizon_days": horizon,
            "train_samples": int(len(train)),
            "validation_samples": int(len(validation)),
            "model_mae": round(model_mae, 4),
            "persistence_mae": round(persistence_mae, 4),
            "validation_origin_start": validation["OriginDate"].min().strftime("%Y-%m-%d"),
            "validation_origin_end": validation["OriginDate"].max().strftime("%Y-%m-%d"),
            "model_file": os.path.basename(model_path),
        })

    metrics = {
        "model_type": "direct_multi_horizon_xgboost",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizons_days": list(HORIZONS),
        "forecast_origin_split_date": split_date.strftime("%Y-%m-%d"),
        "train_targets_before": split_date.strftime("%Y-%m-%d"),
        "validation_origins_from": split_date.strftime("%Y-%m-%d"),
        "validation_fraction": VALIDATION_FRACTION,
        "feature_count": len(features),
        "feature_names": features,
        "leakage_policy": (
            "AQI lags/rolling values end at forecast origin t. Pollutant lags "
            "and pollutant rolling means end at t-1. Target-date calendar "
            "features are known at forecast origin. No target-day pollutant "
            "concentrations are used as inputs."
        ),
        "metrics": horizon_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")
    return metrics


def main():
    daily = load_observed_daily()
    metrics = train_models(daily)
    print(f"Saved metrics: {METRICS_PATH}")
    for result in metrics["metrics"]:
        print(
            f"t+{result['horizon_days']}: "
            f"model MAE={result['model_mae']:.2f}; "
            f"persistence MAE={result['persistence_mae']:.2f} "
            f"(n={result['validation_samples']})"
        )


if __name__ == "__main__":
    main()
