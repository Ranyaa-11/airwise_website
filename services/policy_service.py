"""Policy analysis.

Browse the project's Excel workbooks (Policy.xlsx, DistrictPolicy.xlsx,
Disdata.xlsx). Their column layout is not known to the code, so tables are
shown exactly as stored; the district column is auto-detected.
"""
import os
import threading
import pandas as pd

from . import aqi_service as aqi
from .data_service import DATA_DIR, CITY_ALIASES, CITIES, data_service
from .monitoring_service import monitoring_service

FILES = {"policy": "Policy.xlsx", "district_policy": "DistrictPolicy.xlsx", "district_data": "Disdata.xlsx"}
MAX_ROWS = 500


class PolicyService:
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def _load(self, key):
        path = os.path.join(DATA_DIR, FILES[key])
        if not os.path.exists(path):
            return None
        mtime = os.path.getmtime(path)
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] == mtime:
                return hit[1]
        sheets = pd.read_excel(path, sheet_name=None)
        frames = []
        for name, df in sheets.items():
            df = df.dropna(how="all").dropna(axis=1, how="all")
            df.columns = [str(c).strip() for c in df.columns]
            if len(sheets) > 1:
                df.insert(0, "Sheet", name)
            frames.append(df)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        with self._lock:
            self._cache[key] = (mtime, df)
        return df

    @staticmethod
    def _district_col(df):
        for c in df.columns:
            if any(k in c.lower() for k in ("district", "city", "dist")):
                return c
        return None

    @staticmethod
    def _records(df):
        df = df.head(MAX_ROWS).copy()
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.strftime("%Y-%m-%d")
        df = df.astype(object).where(df.notna(), "")
        return {"columns": list(df.columns), "rows": df.astype(str).values.tolist()}

    @staticmethod
    def _normalize_city_name(name):
        if name is None:
            return ""
        text = str(name).strip()
        if not text:
            return ""
        text = text.replace("_", " ").strip()
        aliases = {
            "Bagalkote": "Bagalkot",
            "Belagaum": "Belgaum",
            "Belagavi": "Belgaum",
            "Bangalore": "Bengaluru",
            "Chikkamagalur": "Chikkamagaluru",
            "Chik Ballapur": "Chikkaballapur",
            "Mysore": "Mysuru",
            "Ramanagar": "Ramanagara",
            "Tumkur": "Tumakuru",
            "Yadgiri": "Yadgir",
            "Mangaluru": "Mangalore",
        }
        return aliases.get(text, text)

    @staticmethod
    def _normalize_pollutant_name(name):
        if name is None:
            return ""
        text = str(name).strip().upper().replace(" ", "")
        aliases = {"PM25": "PM2.5", "PM2_5": "PM2.5", "PM2.5": "PM2.5", "PM10": "PM10",
                   "O3": "OZONE", "OZONE": "OZONE", "NO2": "NO2", "SO2": "SO2", "CO": "CO"}
        return aliases.get(text, text)

    def addresses(self):
        return list(CITIES)

    def districts(self):
        names = set(self.addresses())
        for key in ("district_policy", "district_data"):
            df = self._load(key)
            if df is None:
                continue
            dcol = self._district_col(df)
            if not dcol:
                continue
            for value in df[dcol].dropna().astype(str):
                cleaned = self._normalize_city_name(value)
                if cleaned:
                    names.add(cleaned)
        return sorted(names, key=lambda n: n.lower())

    def policies_for_district(self, district):
        df = self._load("district_policy")
        if df is None:
            return []
        dcol = self._district_col(df)
        if not dcol:
            return []
        district = self._normalize_city_name(district)
        if not district:
            return []
        target = district.lower()
        values = df[dcol].astype(str).str.strip()
        lookup = values.map(lambda s: self._normalize_city_name(s).lower())
        matches = df[lookup == target]
        if matches.empty and district not in self.addresses():
            canonical = self._normalize_city_name(district)
            lookup = values.map(lambda s: self._normalize_city_name(s).lower())
            matches = df[lookup == canonical.lower()]
        policies = [str(v).strip() for v in matches["Policy"].dropna().astype(str)]
        return list(dict.fromkeys(policies))

    def pollutants_for_policy(self, policy):
        df = self._load("policy")
        if df is None:
            return []
        policy = str(policy).strip()
        if not policy:
            return []
        matches = df.loc[df["Policy_Name"].astype(str).str.strip().str.lower() == policy.lower()]
        if matches.empty and "Policy_ID" in df.columns:
            try:
                value = int(policy)
            except ValueError:
                value = None
            if value is not None:
                matches = df.loc[df["Policy_ID"] == value]
        values = []
        for cell in matches["Affected_Pollutants"].dropna().astype(str):
            values.extend([self._normalize_pollutant_name(v) for v in cell.split(",") if v.strip()])
        return list(dict.fromkeys([v for v in values if v]))

    def overview(self):
        out = {}
        for key, fname in FILES.items():
            df = self._load(key)
            out[key] = {"file": fname, "found": df is not None,
                        "rows": 0 if df is None else int(len(df)),
                        "columns": [] if df is None else list(df.columns),
                        "district_column": None if df is None else self._district_col(df)}
        return out

    def table(self, key, district=None, q=None):
        if key not in FILES:
            raise KeyError(key)
        df = self._load(key)
        if df is None:
            return {"found": False, "file": FILES[key]}
        total = len(df)
        dcol = self._district_col(df)
        if district and dcol:
            names = {district.lower()} | {a.lower() for a, c in CITY_ALIASES.items() if c == district}
            col = df[dcol].astype(str).str.strip().str.lower()
            df = df[col.isin(names)]
        if q:
            mask = df.astype(str).apply(lambda s: s.str.contains(q, case=False, na=False, regex=False)).any(axis=1)
            df = df[mask]
        out = self._records(df)
        out.update({"found": True, "file": FILES[key], "district_column": dcol, "matched": int(len(df)),
                    "total": total, "truncated": len(df) > MAX_ROWS,
                    "district_filtered": bool(district and dcol)})
        return out

    def policy_start_date(self, policy_name):
        df = self._load("policy")
        if df is None:
            return None
        if "Policy_Name" not in df.columns:
            return None
        match = df.loc[df["Policy_Name"].astype(str).str.strip().str.lower() == str(policy_name).strip().lower()]
        if match.empty:
            return None
        start = match.iloc[0].get("Start_Date")
        if pd.isna(start):
            return None
        try:
            return pd.to_datetime(start).strftime("%Y-%m-%d")
        except Exception:
            return str(start)

    def analyze(self, district, policy_name, window_months=1):
        district = self._normalize_city_name(district)
        if not district:
            raise ValueError("District is required")
        try:
            months = int(window_months)
        except (TypeError, ValueError):
            months = 1
        months = max(1, min(6, months))

        affected = self.pollutants_for_policy(policy_name)
        if not affected:
            raise ValueError(f"Policy '{policy_name}' has no pollutant list in Policy.xlsx")

        policy_start = self.policy_start_date(policy_name)

        df = self._load("district_data")
        if df is None:
            raise FileNotFoundError("Disdata.xlsx is missing from data/")
        city_col = self._district_col(df) or "City"
        if city_col not in df.columns:
            raise ValueError("Disdata.xlsx does not contain a district column")

        city_values = df[city_col].astype(str).str.strip().map(self._normalize_city_name)
        district_matches = city_values.str.lower() == district.lower()
        city_df = df.loc[district_matches].copy()
        if city_df.empty:
            raise ValueError(f"No district data found for '{district}'")

        city_df["Date"] = pd.to_datetime(city_df["Date"], errors="coerce")
        city_df = city_df.dropna(subset=["Date"]).sort_values("Date")
        if city_df.empty:
            raise ValueError(f"No valid dates found for '{district}'")

        analysis_start = None
        analysis_end = city_df["Date"].max()
        if policy_start:
            analysis_start = pd.to_datetime(policy_start)
            before_end = analysis_start - pd.Timedelta(days=1)
            before_start = before_end - pd.DateOffset(months=months)
            after_start = analysis_start
            after_end = min(analysis_end, after_start + pd.DateOffset(months=months))
        else:
            analysis_start = city_df["Date"].min()
            before_start = city_df["Date"].min()
            before_end = city_df["Date"].max()
            after_start = city_df["Date"].min()
            after_end = city_df["Date"].max()

        before_df = city_df[(city_df["Date"] >= before_start) & (city_df["Date"] <= before_end)] if policy_start else city_df
        after_df = city_df[(city_df["Date"] >= after_start) & (city_df["Date"] <= after_end)] if policy_start else city_df

        if policy_start and (before_df.empty or after_df.empty):
            before_df = city_df[city_df["Date"] <= (pd.to_datetime(policy_start) - pd.Timedelta(days=1))]
            after_df = city_df[city_df["Date"] >= pd.to_datetime(policy_start)]

        rows = []
        for pollutant in affected:
            if pollutant not in city_df.columns:
                continue
            before_series = pd.to_numeric(before_df[pollutant], errors="coerce")
            after_series = pd.to_numeric(after_df[pollutant], errors="coerce")
            before_vals = before_series.dropna()
            after_vals = after_series.dropna()
            if before_vals.empty or after_vals.empty:
                continue
            before = float(before_vals.mean())
            after = float(after_vals.mean())
            change_pct = ((before - after) / before) * 100.0 if before else 0.0
            impact = "No Impact"
            if change_pct >= 20:
                impact = "Improved"
            elif change_pct >= 5:
                impact = "Low Impact"
            rows.append({
                "pollutant": pollutant,
                "before": round(before, 2),
                "after": round(after, 2),
                "change_pct": round(change_pct, 1),
                "impact": impact,
                "policy_start_date": policy_start,
            })

        if not rows:
            raise ValueError(f"No pollutant data found for {district} in the selected window")

        avg_before = sum(r["before"] for r in rows) / len(rows)
        avg_after = sum(r["after"] for r in rows) / len(rows)
        overall_change = ((avg_before - avg_after) / avg_before) * 100.0 if avg_before else 0.0
        if overall_change >= 20:
            status = "Improved"
        elif overall_change >= 5:
            status = "Low Impact"
        else:
            status = "No Impact"

        summary = (
            f"For {district}, {policy_name} shows {overall_change:.1f}% average improvement across "
            f"{', '.join(r['pollutant'] for r in rows[:3])} over the selected {months}-month analysis window."
        )
        recommendation = (
            f"{policy_name} was implemented on {policy_start or 'unknown date'} and the observed impact is {status.lower()} for {district}. "
            "Review pollutant-specific trends and continue enforcement where reductions remain limited."
        )

        return {
            "city": district,
            "district": district,
            "policy": policy_name,
            "implementation_date": policy_start,
            "window_months": months,
            "periods": {
                "before_start": before_df["Date"].min().strftime("%Y-%m-%d"),
                "before_end": before_df["Date"].max().strftime("%Y-%m-%d"),
                "after_start": after_df["Date"].min().strftime("%Y-%m-%d"),
                "after_end": after_df["Date"].max().strftime("%Y-%m-%d"),
            },
            "affected_pollutants": [r["pollutant"] for r in rows],
            "results": rows,
            "status": status,
            "summary": summary,
            "recommendation": recommendation,
            "before_avg": round(avg_before, 2),
            "after_avg": round(avg_after, 2),
            "overall_change_pct": round(overall_change, 1),
        }



policy_service = PolicyService()
