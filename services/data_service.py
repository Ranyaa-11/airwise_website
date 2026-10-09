"""Load and impute station history, build daily CPCB AQI, and serve analytics."""
import os
import json
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import aqi_service as aqi
from .freshness import IST, freshness, source_age_minutes

IST = ZoneInfo("Asia/Kolkata")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
DAILY_CACHE = os.path.join(CACHE_DIR, "daily.csv")

# Coordinates from the notebook (CITY_COORDS)
CITY_COORDS = {
    "Bagalkot": [16.1725, 75.6550], "Belgaum": [15.8497, 74.4977],
    "Bengaluru": [12.9173, 77.6228], "Chamarajanagar": [11.9231, 76.9395],
    "Chikkaballapur": [13.4350, 77.7315], "Chikkamagaluru": [13.3161, 75.7720],
    "Dharwad": [15.4589, 75.0078], "Kalaburagi": [17.3297, 76.8343],
    "Madikeri": [12.4244, 75.7382], "Mangalore": [12.9141, 74.8560],
    "Mysuru": [12.2958, 76.6394], "Ramanagara": [12.7200, 77.2800],
    "Tumakuru": [13.3409, 77.1010], "Yadgir": [16.7704, 77.1378],
}
CITIES = list(CITY_COORDS.keys())
CITY_ALIASES = {"Belagavi": "Belgaum", "Mangaluru": "Mangalore", "Tumkur": "Tumakuru",
                "Yadgiri": "Yadgir", "Chik Ballapur": "Chikkaballapur"}


def resolve_city(name):
    if not name:
        return None
    name = CITY_ALIASES.get(name.strip(), name.strip())
    for c in CITIES:
        if c.lower() == name.lower():
            return c
    return None


def load_raw(data_dir=DATA_DIR, verbose=False):
    """Notebook Cell 1: read <City>.xlsx, clean, return hourly/raw rows."""
    frames, missing = [], []
    for city in CITIES:
        path = os.path.join(data_dir, f"{city}.xlsx")
        if not os.path.exists(path):
            missing.append(city)
            continue
        df = pd.read_excel(path)
        df["City"] = city
        if "From Date" not in df.columns:
            raise ValueError(f"{city}.xlsx has no 'From Date' column (found: {list(df.columns)})")
        df["From Date"] = pd.to_datetime(df["From Date"], dayfirst=True, errors="coerce")
        for col in aqi.POLLUTANTS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df.loc[df[col] < 0, col] = np.nan
            else:
                df[col] = np.nan
        df = df.dropna(subset=["From Date"]).drop_duplicates(subset=["From Date", "City"], keep="first")
        frames.append(df[["City", "From Date"] + aqi.POLLUTANTS])
        if verbose:
            print(f"  loaded {city:<16} {len(df):>7,} rows")
    if not frames:
        raise FileNotFoundError(f"No city Excel files found in {data_dir}")
    raw = pd.concat(frames, ignore_index=True)
    # CO normalisation (notebook): if median looks like µg/m³ convert to mg/m³
    if raw["CO"].notna().any() and raw["CO"].median() > 100:
        raw["CO"] = raw["CO"] / 1000.0
    for pollutant in aqi.POLLUTANTS:
        raw[f"_missing_{pollutant}"] = raw[pollutant].isna()
    raw = colab_impute(raw)
    return raw.sort_values(["City", "From Date"]).reset_index(drop=True), missing


def colab_impute(raw, P=("PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE")):
    """Fill pollutant gaps using the same time/weekday/city sequence as Colab."""
    parts = []
    for city, group in raw.groupby("City", sort=False):
        group = group.sort_values("From Date").set_index("From Date")
        for pollutant in P:
            group[pollutant] = group[pollutant].interpolate(
                method="time", limit=7, limit_direction="both"
            )
        group = group.reset_index()
        group["City"] = city
        parts.append(group)

    result = pd.concat(parts, ignore_index=True)
    result["_wd"] = result["From Date"].dt.dayofweek
    for pollutant in P:
        result[pollutant] = result[pollutant].fillna(
            result.groupby(["City", "_wd"])[pollutant].transform("mean")
        )
    for pollutant in P:
        result[pollutant] = result[pollutant].fillna(
            result.groupby("City")[pollutant].transform("mean")
        )
    return result.drop(columns="_wd")


def build_daily(raw):
    """Notebook Cell 2: daily mean per city + CPCB AQI."""
    raw = raw.copy()
    raw["Date"] = raw["From Date"].dt.floor("D")
    daily = raw.groupby(["City", "Date"], as_index=False)[aqi.POLLUTANTS].mean()
    for pollutant in aqi.POLLUTANTS:
        missing_column = f"_missing_{pollutant}"
        if missing_column not in raw.columns:
            raw[missing_column] = raw[pollutant].isna()
        missing_by_day = raw.groupby(["City", "Date"])[missing_column].any()
        daily[f"_missing_{pollutant}"] = [
            bool(missing_by_day.loc[(row.City, row.Date)])
            for row in daily.itertuples()
        ]

    def row_aqi(r):
        return aqi.calculate_aqi({p: r[p] for p in aqi.POLLUTANTS})["aqi"]

    daily["AQI"] = daily.apply(row_aqi, axis=1)
    daily["Category"] = daily["AQI"].apply(aqi.category)
    daily["possible_outlier"] = False
    for _, city_rows in daily.groupby("City", sort=False):
        ordered = city_rows.sort_values("Date")
        # Compare isolated spikes with AQI observations within three calendar days on each side.
        neighbors = []
        for offset in (1, 2, 3):
            previous_date = ordered["Date"].shift(offset)
            next_date = ordered["Date"].shift(-offset)
            neighbors.append(ordered["AQI"].shift(offset).where(
                ordered["Date"].sub(previous_date).le(pd.Timedelta(days=3))
            ))
            neighbors.append(ordered["AQI"].shift(-offset).where(
                next_date.sub(ordered["Date"]).le(pd.Timedelta(days=3))
            ))
        neighbors = pd.concat(neighbors, axis=1)
        neighbor_median = neighbors.median(axis=1)
        previous_day = ordered["Date"].diff().eq(pd.Timedelta(days=1))
        next_day = ordered["Date"].shift(-1).sub(ordered["Date"]).eq(pd.Timedelta(days=1))
        outliers = (
            previous_day
            & next_day
            & ordered["AQI"].gt(neighbor_median * 3)
        )
        daily.loc[ordered.index, "possible_outlier"] = outliers.to_numpy()
    return daily.sort_values(["City", "Date"]).reset_index(drop=True)


class DataService:
    def __init__(self):
        self._lock = threading.Lock()
        self._daily = None
        self._source_mtimes = None
        self._source_check_at = None
        self._reload_from_sources = False

    def _check_excel_changes_locked(self):
        now = time.monotonic()
        if self._source_check_at is not None and now - self._source_check_at < 60:
            return

        source_mtimes = {}
        for city in CITIES:
            path = os.path.join(DATA_DIR, f"{city}.xlsx")
            try:
                source_mtimes[path] = os.stat(path).st_mtime_ns
            except FileNotFoundError:
                continue

        if self._source_mtimes is not None and source_mtimes != self._source_mtimes:
            self._daily = None
            self._reload_from_sources = True
        self._source_mtimes = source_mtimes
        self._source_check_at = now

    @staticmethod
    def _previous_day_cutoff():
        return pd.Timestamp(datetime.now(IST).date()) - pd.Timedelta(days=1)

    def _valid_history(self, d):
        if d.empty:
            return d
        cutoff = self._previous_day_cutoff()
        valid = d[d["Date"] <= cutoff]
        return valid if not valid.empty else d

    # ---- loading -------------------------------------------------------
    def daily(self):
        with self._lock:
            self._check_excel_changes_locked()
            if self._daily is None:
                if not self._reload_from_sources and os.path.exists(DAILY_CACHE):
                    df = pd.read_csv(DAILY_CACHE, parse_dates=["Date"])
                    missing_columns = [
                        f"_missing_{pollutant}" for pollutant in aqi.POLLUTANTS
                    ]
                    if "possible_outlier" not in df.columns or not set(missing_columns).issubset(df.columns):
                        raw, _ = load_raw()
                        df = build_daily(raw)
                else:
                    raw, _ = load_raw()
                    df = build_daily(raw)
                self._daily = df
                self._reload_from_sources = False
            return self._daily

    def reload(self):
        with self._lock:
            self._daily = None
            self._reload_from_sources = True

    def available(self):
        try:
            return len(self.daily()) > 0
        except Exception:
            return False

    def city_daily(self, city):
        d = self.daily()
        d = d[d["City"] == city].sort_values("Date").reset_index(drop=True)
        return self._valid_history(d)

    # ---- model inputs ----------------------------------------------------
    def aqi_log_history(self, city, n=60):
        """log1p(AQI) of the most recent days that have a valid AQI."""
        d = self.city_daily(city)
        d = d[d["AQI"].notna()]
        return list(np.log1p(d["AQI"].clip(lower=0).values[-n:])), (d["Date"].max() if len(d) else None)

    def latest_pollutants(self, city):
        """Pollutants from the most recent day with a valid historical AQI."""
        d = self.city_daily(city)
        d = d[d["AQI"].notna()]
        if d.empty:
            return None, None
        r = d.iloc[-1]
        return {p: float(r[p]) for p in aqi.POLLUTANTS}, r["Date"]

    def latest_station_reading(self, city):
        """Return the latest daily AQI, retaining an outlier flag for display."""
        d = self.city_daily(city)
        d = d[d["AQI"].notna()]
        if d.empty:
            return None
        row = d.iloc[-1]
        source_timestamp = pd.Timestamp(row["Date"]).to_pydatetime()
        if source_timestamp.tzinfo is None:
            source_timestamp = source_timestamp.replace(tzinfo=IST)
        else:
            source_timestamp = source_timestamp.astimezone(IST)
        return {
            "date": row["Date"].strftime("%Y-%m-%d"),
            "source_timestamp": source_timestamp.isoformat(),
            "source_time": source_timestamp.strftime("%Y-%m-%d %H:%M:%S IST"),
            "age_minutes": source_age_minutes(source_timestamp),
            "aqi": int(row["AQI"]),
            "category": row["Category"],
            "possible_outlier": bool(row.get("possible_outlier", False)),
            "pollutants": [
                {
                    "key": pollutant,
                    "label": aqi.LABELS[pollutant],
                    "value": (
                        None
                        if bool(row.get(f"_missing_{pollutant}", False))
                        else float(row[pollutant])
                    ),
                    "unit": aqi.UNITS[pollutant],
                    "filled_in": bool(row.get(f"_missing_{pollutant}", False)),
                }
                for pollutant in aqi.POLLUTANTS
            ],
            "filled_in": [
                pollutant for pollutant in aqi.POLLUTANTS
                if bool(row.get(f"_missing_{pollutant}", False))
            ],
            **freshness(row["Date"]),
        }

    def data_through(self, city):
        d = self.city_daily(city)
        d = d[d["AQI"].notna()]
        return None if d.empty else d["Date"].max()

    # ---- analytics for the UI -------------------------------------------
    def trend(self, city, days=90):
        d = self.city_daily(city)
        d = d[d["AQI"].notna()].tail(days)
        return {"dates": d["Date"].dt.strftime("%Y-%m-%d").tolist(),
                "aqi": [int(x) for x in d["AQI"]],
                "possible_outlier": d.get("possible_outlier", pd.Series(False, index=d.index)).astype(bool).tolist(),
                "categories": d["Category"].tolist()}

    def monthly_profile(self, city):
        d = self.city_daily(city)
        d = d[d["AQI"].notna()]
        if d.empty:
            return {"months": [], "mean_aqi": [], "n_days": 0, "last_date": None}
        m = d.groupby(d["Date"].dt.month)["AQI"].mean().round(1)
        return {
            "months": [int(i) for i in m.index],
            "mean_aqi": m.tolist(),
            "n_days": int(len(d)),
            "last_date": d["Date"].max().strftime("%Y-%m-%d"),
        }

    def baseline_means(self, city, days=365):
        """Mean pollutant concentrations over the last `days` of available history."""
        d = self.city_daily(city)
        if d.empty:
            return None
        d = d[d["Date"] >= d["Date"].max() - pd.Timedelta(days=days)]
        out = {p: (None if d[p].dropna().empty else float(d[p].mean())) for p in aqi.POLLUTANTS}
        return out

    def summary(self):
        d = self.daily()
        out = {}
        for c in CITIES:
            cd = d[(d["City"] == c) & d["AQI"].notna()]
            out[c] = None if cd.empty else {
                "first": cd["Date"].min().strftime("%Y-%m-%d"),
                "last": cd["Date"].max().strftime("%Y-%m-%d"),
                "days": int(len(cd)),
                "mean_aqi": round(float(cd["AQI"].mean()), 1)}
        return out


data_service = DataService()
