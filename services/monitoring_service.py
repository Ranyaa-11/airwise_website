"""Real-time monitoring: live API (same endpoint/fields as the notebook) +
optional hardware sensors posting readings to /api/sensors/reading.

Every successful live fetch is logged to SQLite so the site builds its own
real observation history (shown as the "recorded by AIRWISE" chart)."""
import os
import json
import time
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests

from . import aqi_service as aqi
from .data_service import CITY_COORDS, DATA_DIR
from .freshness import source_age_minutes

IST = ZoneInfo("Asia/Kolkata")
DB_PATH = os.path.join(DATA_DIR, "airwise.db")


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


class MonitoringService:
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        self._init_db()

    # ---- sqlite ------------------------------------------------------------
    def _db(self):
        con = sqlite3.connect(DB_PATH, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with self._db() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS live_log(
                id INTEGER PRIMARY KEY, city TEXT, ts_utc TEXT, api_time TEXT,
                aqi INTEGER, dominant TEXT, pollutants TEXT)""")
            con.execute("""CREATE TABLE IF NOT EXISTS sensor_readings(
                id INTEGER PRIMARY KEY, city TEXT, device_id TEXT, ts_utc TEXT,
                pollutants TEXT)""")
            con.execute("CREATE INDEX IF NOT EXISTS ix_live ON live_log(city, ts_utc)")
            con.execute("CREATE INDEX IF NOT EXISTS ix_sens ON sensor_readings(city, ts_utc)")

    @staticmethod
    def _parse_timestamp(value):
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=IST) if parsed.tzinfo is None else parsed.astimezone(IST)

    @staticmethod
    def _display_timestamp(value):
        parsed = MonitoringService._parse_timestamp(value)
        return parsed.strftime("%Y-%m-%d %H:%M:%S IST") if parsed else None

    def _last_success_time(self, city):
        with self._db() as con:
            row = con.execute(
                "SELECT ts_utc FROM live_log WHERE city=? AND ts_utc IS NOT NULL ORDER BY id DESC LIMIT 1",
                (city,),
            ).fetchone()
        return self._display_timestamp(row["ts_utc"]) if row else None

    def last_success_time(self, city):
        return self._last_success_time(city)

    def _feed_failure(self, city, message):
        last_success = self._last_success_time(city)
        return {
            "ok": False,
            "error": f"Estimate unavailable (last success: {last_success or '--'})",
            "detail": message,
            "last_success": last_success,
        }

    # ---- live API ---------------------------------------------------------------
    def fetch_api(self, city):
        """Raw pollutant values from the live API; cached per city."""
        ttl = min(600, max(0, _env_int("CACHE_TTL_SECONDS", 600)))
        with self._lock:
            hit = self._cache.get(city)
            if hit and time.time() - hit["t"] < ttl:
                return hit["data"]
        lat, lon = CITY_COORDS[city]
        url = os.getenv("CPCB_API_URL", "https://app.cpcbccr.com/api/location-data")
        try:
            r = requests.get(url, params={"lat": lat, "lon": lon, "fresh": "1"}, timeout=25)
            if r.status_code != 200:
                return self._feed_failure(city, f"Live API returned HTTP {r.status_code}")
            payload = r.json()
            aq = payload.get("airQuality", {}) or {}
            cur = aq.get("current", {}) or {}
            co_ug = cur.get("carbon_monoxide")
            vals = {
                "PM2.5": cur.get("pm2_5"), "PM10": cur.get("pm10"),
                "NO2": cur.get("nitrogen_dioxide"), "SO2": cur.get("sulphur_dioxide"),
                "CO": (float(co_ug) / 1000.0) if co_ug is not None else None,   # µg/m³ -> mg/m³
                "OZONE": cur.get("ozone"),
            }
            data = {"ok": True, "values": vals, "api_time": cur.get("time"),
                    "us_aqi": cur.get("us_aqi"),
                    "grid_lat": aq.get("latitude"), "grid_lon": aq.get("longitude"),
                    "fetched_at": datetime.now(timezone.utc).isoformat()}
        except Exception as e:  # network, JSON, timeout
            return self._feed_failure(city, f"Live API unreachable: {e}")
        with self._lock:
            self._cache[city] = {"t": time.time(), "data": data}
        self._log_live(city, data)
        return data

    def _log_live(self, city, data):
        try:
            res = aqi.calculate_aqi(data["values"])
            with self._db() as con:
                con.execute("INSERT INTO live_log(city, ts_utc, api_time, aqi, dominant, pollutants) VALUES(?,?,?,?,?,?)",
                            (city, datetime.now(timezone.utc).isoformat(), data.get("api_time"),
                             res["aqi"], res["dominant"], json.dumps(data["values"])))
        except Exception:
            pass

    # ---- sensors -------------------------------------------------------------------
    def ingest_sensor(self, city, device_id, values):
        clean = {}
        for p in aqi.POLLUTANTS:
            v = aqi._num(values.get(p, values.get(p.lower())))
            if v is not None and v >= 0:
                clean[p] = v
        if not clean:
            raise ValueError("No valid pollutant values in payload")
        with self._db() as con:
            con.execute("INSERT INTO sensor_readings(city, device_id, ts_utc, pollutants) VALUES(?,?,?,?)",
                        (city, device_id or "unknown", datetime.now(timezone.utc).isoformat(), json.dumps(clean)))
        with self._lock:
            self._cache.pop(city, None)
        return clean

    def latest_sensor(self, city):
        max_age = _env_int("SENSOR_MAX_AGE_MIN", 60)
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=max_age)).isoformat()
        with self._db() as con:
            rows = con.execute("SELECT * FROM sensor_readings WHERE city=? AND ts_utc>=? ORDER BY ts_utc DESC LIMIT 50",
                               (city, cutoff)).fetchall()
        if not rows:
            return None
        merged, ts, devices = {}, rows[0]["ts_utc"], set()
        for r in rows:                      # newest reading per pollutant wins
            devices.add(r["device_id"])
            for p, v in json.loads(r["pollutants"]).items():
                merged.setdefault(p, v)
        return {"values": merged, "ts_utc": ts, "devices": sorted(devices)}

    def sensor_status(self):
        with self._db() as con:
            n = con.execute("SELECT COUNT(*) c FROM sensor_readings").fetchone()["c"]
            last = con.execute("SELECT city, device_id, ts_utc FROM sensor_readings ORDER BY ts_utc DESC LIMIT 1").fetchone()
        return {"total_readings": n, "last": dict(last) if last else None}

    # ---- merged current reading --------------------------------------------------------
    def current(self, city):
        api = self.fetch_api(city)
        sensor = self.latest_sensor(city)
        if not api.get("ok") and not sensor:
            return {
                "ok": False,
                "city": city,
                "error": api.get("error", "Estimate unavailable (last success: --)"),
                "api_ok": False,
                "api_error": api.get("detail"),
                "last_success": api.get("last_success"),
            }

        values, origin = {}, {}
        for p in aqi.POLLUTANTS:
            if api.get("ok") and api["values"].get(p) is not None:
                values[p], origin[p] = float(api["values"][p]), "api"
            if sensor and p in sensor["values"]:
                values[p], origin[p] = float(sensor["values"][p]), "sensor"

        res = aqi.calculate_aqi(values)
        cat = aqi.category(res["aqi"])
        now = datetime.now(IST)
        api_timestamp = (
            self._parse_timestamp(api.get("api_time"))
            if api.get("ok") and any(origin.get(p) == "api" for p in aqi.POLLUTANTS)
            else None
        )
        sensor_timestamp = (
            self._parse_timestamp(sensor["ts_utc"])
            if sensor and any(origin.get(p) == "sensor" for p in aqi.POLLUTANTS)
            else None
        )
        source_timestamps = [
            stamp for stamp in (api_timestamp, sensor_timestamp) if stamp is not None
        ]
        api_age_minutes = source_age_minutes(api_timestamp, now)
        sensor_age_minutes = source_age_minutes(sensor_timestamp, now)
        source_ages = [
            age for age in (api_age_minutes, sensor_age_minutes) if age is not None
        ]
        estimate_age_minutes = max(source_ages) if source_ages else None
        estimate_time = (
            min(source_timestamps).strftime("%Y-%m-%d %H:%M:%S IST")
            if source_timestamps else None
        )
        source_ages_by_name = {
            "cpcbccr": {
                "timestamp": api_timestamp.strftime("%Y-%m-%d %H:%M:%S IST"),
                "age_minutes": api_age_minutes,
            } if api_timestamp else None,
            "sensor": {
                "timestamp": sensor_timestamp.strftime("%Y-%m-%d %H:%M:%S IST"),
                "age_minutes": sensor_age_minutes,
            } if sensor_timestamp else None,
        }
        pollutants = []
        for p in aqi.POLLUTANTS:
            v = values.get(p)
            pollutants.append({"key": p, "label": aqi.LABELS[p], "value": None if v is None else round(v, 2 if p != "CO" else 2),
                               "unit": aqi.UNITS[p], "subindex": res["subindices"].get(p),
                               "source": origin.get(p), "info": aqi.POLLUTANT_INFO[p]})
        sources = sorted(set(origin.values()))
        return {
            "ok": True, "city": city,
            "aqi": res["aqi"], "category": cat, "color": aqi.color(res["aqi"]),
            "dominant": res["dominant"], "dominant_info": aqi.POLLUTANT_INFO.get(res["dominant"]),
            "health": aqi.HEALTH.get(cat), "pollutants": pollutants,
            "missing": [aqi.LABELS[p] for p in aqi.POLLUTANTS if values.get(p) is None],
            "values": values,
            "data_sources": sources,
            "api_ok": bool(api.get("ok")), "api_error": None if api.get("ok") else api.get("error"),
            "api_time": api.get("api_time"), "us_aqi": api.get("us_aqi"),
            "estimate_time": estimate_time,
            "estimate_age_minutes": None if estimate_age_minutes is None else round(estimate_age_minutes, 1),
            "estimate_old": estimate_age_minutes is not None and estimate_age_minutes > 180,
            "source_ages": source_ages_by_name,
            "feed_warning": None if api.get("ok") else api.get("error"),
            "sensor": None if not sensor else {"devices": sensor["devices"], "ts_utc": sensor["ts_utc"]},
            "grid": [api.get("grid_lat"), api.get("grid_lon")] if api.get("ok") else None,
            "coords": CITY_COORDS[city],
            "fetched_at": self._display_timestamp(api.get("fetched_at")) if api.get("ok") else None,
        }

    def live_log(self, city, hours=48):
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._db() as con:
            rows = con.execute("SELECT ts_utc, aqi FROM live_log WHERE city=? AND ts_utc>=? AND aqi IS NOT NULL ORDER BY ts_utc",
                               (city, cutoff)).fetchall()
        out = []
        for r in rows:
            t = datetime.fromisoformat(r["ts_utc"]).astimezone(IST)
            out.append({"t": t.strftime("%d %b %H:%M"), "aqi": r["aqi"]})
        return out


monitoring_service = MonitoringService()
