# AIRWISE

## Run
1. Put the 14 city workbooks (`Bagalkot.xlsx` … `Yadgir.xlsx`) plus `Policy.xlsx`, `DistrictPolicy.xlsx`, `Disdata.xlsx` in `data/`.
2. `pip install -r requirements.txt`
3. `python prepare_data.py`   – builds `data/cache/daily.csv`, validates model files
4. `python app.py`  → http://localhost:5000

## Daily station-data ingestion
Place one fresh station export for each district in `data/incoming/`, named
`<City>.csv` or `<City>.xlsx`. Each export must contain `From Date` plus
`PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, and `OZONE` columns. Then run
`python ingest_station_data.py` (or schedule it daily). The importer appends
new dates, replaces duplicate dates with the incoming row, and refuses to
update any workbook unless the latest date for every district is no more than
two days old. It imports supplied exports; it does not fetch station data.
Run `python prepare_data.py` after ingestion to rebuild the daily AQI cache.

## Prediction model
The prediction page reads the production feature list from
`models/airwise_production_features.pkl`. It prefers the portable
`models/airwise_xgboost_production_model.json` model and falls back to the
legacy `models/airwise_xgboost_production_model.pkl` pickle when no JSON model
is present. If the legacy pickle cannot be loaded by the installed XGBoost
version, export the fitted model in Colab with
`BEST_MODEL.save_model("airwise_xgboost_production_model.json")`,
copy the resulting JSON file into `models/`, then restart the website and run
`python prepare_data.py`.

## Direct multi-horizon AQI training
Run `python train_direct_forecast.py` to train independent XGBoost models for
t+1 through t+7 from observed daily station data. Pollutant lag/rolling inputs
end at t-1; AQI history may include AQI(t), and target-date calendar features
are allowed because they are known at forecast time. A chronological holdout
reports each horizon's MAE beside the persistence baseline. Model JSON files
and `models/direct_aqi_metrics.json` are written for the app; validation
metrics are available at `/api/predict/direct-metrics` and on the Predictions
page. This direct-model evaluation is informational and does not replace the
existing persistence outlook serving path.

## Sensors
POST JSON to `/api/sensors/reading` with header `X-API-Key: <SENSOR_API_KEY from .env>`:
`{"district":"Bengaluru","device_id":"node-1","PM2.5":41.2,"PM10":80,"NO2":22,"SO2":5,"CO":0.8,"OZONE":35}`
(CO in mg/m³, others µg/m³; any subset of pollutants is accepted). A reading overrides the live API value for that
pollutant for `SENSOR_MAX_AGE_MIN` minutes.

## Check status
`/api/health` shows history coverage, whether the model loaded, which workbooks were found and sensor activity.
