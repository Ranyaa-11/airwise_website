"""Run once (and again whenever you add new rows to the city Excel files):

    python prepare_data.py

Reads data/<City>.xlsx, applies the notebook pollutant imputation, builds the
daily CPCB AQI table, writes data/cache/daily.csv and validates model files.
"""
import os
import sys

from services import data_service as ds
from services.prediction_service import prediction_service


def main():
    print("=" * 70)
    print("AIRWISE - PREPARE DATA")
    print("=" * 70)
    raw, missing = ds.load_raw(verbose=True)
    if missing:
        print("\nMISSING city files:", ", ".join(missing))

    daily = ds.build_daily(raw)
    os.makedirs(ds.CACHE_DIR, exist_ok=True)
    daily.to_csv(ds.DAILY_CACHE, index=False)

    print("\nDaily records :", f"{len(daily):,}")
    print("Valid AQI days:", int(daily['AQI'].notna().sum()))
    print("\nPer-city coverage:")
    print(f"{'City':<16}{'first':<12}{'last':<12}{'days':>6}{'mean AQI':>10}")
    for c in ds.CITIES:
        cd = daily[(daily.City == c) & daily.AQI.notna()]
        if cd.empty:
            print(f"{c:<16}-- no valid AQI days --")
        else:
            print(f"{c:<16}{cd.Date.min():%Y-%m-%d}  {cd.Date.max():%Y-%m-%d}  {len(cd):>6}{cd.AQI.mean():>10.1f}")

    # ---- model files ----
    print("\nModel files:")
    model_status = prediction_service.status()
    print("  features:", model_status["features"], "(expected 64)")
    if model_status["loaded"]:
        print("  model   : loaded OK")
    else:
        print("  model   : FAILED to load ->", model_status["error"])
        sys.exit(1)

    # ---- policy workbooks ----
    print("\nPolicy / district workbooks:")
    for f in ("Policy.xlsx", "DistrictPolicy.xlsx", "Disdata.xlsx"):
        print(f"  {f:<22}", "found" if os.path.exists(os.path.join(ds.DATA_DIR, f)) else "MISSING")
    print("\nDone. Start the site with:  python app.py")


if __name__ == "__main__":
    main()
