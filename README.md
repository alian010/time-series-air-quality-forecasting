Time-Series Forecasting of Carbon Monoxide (CO) and Nitrogen Dioxide (NO₂)

Forecast hourly concentrations of CO(GT) and NO₂(GT) from historical air-quality data.
This repo contains a complete, student-friendly notebook with short English comments that walks through: preprocessing → EDA → feature engineering → SARIMAX & ML models → evaluation → visualization → insights.

Dataset (public CSV):
https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/airquality.csv

Table of Contents

Project Overview

Repository Structure

Environment & Setup

Data & Preprocessing

Exploratory Data Analysis (EDA)

Feature Engineering

Forecasting Models

Evaluation

Visualization & Insights

Troubleshooting (SARIMAX exogenous shape)

Next Steps

License

Project Overview

Goal: Build reliable short-term forecasts (e.g., next 24h) for CO(GT) and NO₂(GT) and identify patterns that inform air-quality management.

Approach:

Parse datetimes, resample to hourly means.

Interpolate missing values; winsorize targets to cap outliers.

Engineer lag features and time stamps (hour, day of week, month).

Train SARIMAX (classic time-series) and RandomForestRegressor (lag-based ML).

Evaluate with MAE / RMSE / MAPE and plot actual vs predicted.

Repository Structure
.
├─ notebooks/
│  └─ Time-Series Forecasting of Carbon Monoxide and Nitrogen Dioxide Levels.ipynb
├─ models/                  # (optional) saved models
├─ data/                    # (optional) local cache
├─ requirements.txt
├─ .gitignore
└─ README.md


For coursework, it’s fine to keep everything inside notebooks/.

Environment & Setup
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt


Suggested versions:
pandas>=2.0, numpy>=1.24, matplotlib>=3.7, seaborn>=0.12, scikit-learn>=1.3, statsmodels>=0.13

Data & Preprocessing

Columns used (if available): Date, Time, CO(GT), NO2(GT), NOx(GT), C6H6(GT), T (temp), RH (humidity), AH (absolute humidity), and PT08.* sensor channels.

Steps:

Datetime parsing: datetime = to_datetime(Date + ' ' + Time, dayfirst=True) → set as index → sort.

Resampling: hourly means ('H'). You can switch to daily ('D') if desired.

Missing values: time interpolation + forward/back fill.

Outliers: winsorize CO(GT) and NO₂(GT) at 0.5% / 99.5% quantiles.

Exploratory Data Analysis (EDA)

Trends: line plots for CO(GT) and NO₂(GT) across the full time range.

Seasonality: average by hour of day and day of week (diurnal/weekly cycles).

Correlations: heatmap of targets with meteorology (T, RH, AH) and sensor channels.

Feature Engineering

Time features: hour, dow (0–6), month.

Lag features: for each target, lags at [1, 2, 3, 6, 12, 24] hours (extend as needed).

External predictors: meteorology (T, RH, AH) and sensor channels for ML models (and optional SARIMAX exog).

Forecasting Models

SARIMAX (per target):

Baseline order=(1,0,1) with seasonal weekly-ish pattern for hourly data (seasonal_order=(1,0,1,24)); adjust by AIC.

Optional exogenous regressors: T, RH, AH (and/or sensors).

RandomForestRegressor (lagged ML):

Train on engineered lags + time + meteorology; simple and robust baseline.

Train/validation split: last ~15% (or at least 168 hours) used as validation window.

Evaluation

Metrics:

MAE (mean absolute error)

RMSE (root mean squared error)

MAPE (mean absolute percentage error; small epsilon protects near-zero targets)

Visuals: overlay plots of validation actual vs predicted for both models and both targets.

Visualization & Insights

Forecast plots: history (last week) + next 24h forecast (when SARIMAX exog is correctly shaped; see troubleshooting).

Example insights (generic):

High CO periods often align with traffic peaks; discourage idling and promote public transit during rush hours.

Elevated NO₂ is linked to combustion sources; adjust urban traffic flow and encourage low-NOx technologies.

Meteorology affects dispersion—during low wind / high RH, anticipate higher concentrations; plan alerts accordingly.

Troubleshooting (SARIMAX exogenous shape)

If you see:

Future forecast failed for CO(GT): Provided exogenous values are not of the appropriate shape. Required (2544, 3), got (24, 3).
Future forecast failed for NO2(GT): Provided exogenous values are not of the appropriate shape. Required (2544, 3), got (24, 3).


Cause:
SARIMAXResults.predict(start=..., end=..., exog=...) expects exog for the entire span from the model’s start index through the forecast horizon (in-sample + out-of-sample), not just future rows. Passing only (steps, n_exog) causes a shape mismatch.

Fix (recommended): use get_forecast with steps and future exog only

# Assume you've already fitted model: res = SARIMAX(...).fit()
# Build future index (hourly)
steps = 24
future_index = pd.date_range(df_h.index[-1] + pd.Timedelta(hours=1), periods=steps, freq="H")

# Build future exog with SAME columns & order used during fit
exog_cols = ["T", "RH", "AH"]  # exactly what you used while fitting
last_exog = df_h[exog_cols].iloc[-1:].copy()           # naive hold
exog_future = pd.concat([last_exog] * steps, ignore_index=True)
exog_future.index = future_index

# Correct API for out-of-sample forecasts:
fc_res = res.get_forecast(steps=steps, exog=exog_future)
fc = fc_res.predicted_mean
fc.index = future_index


Alternative (if you prefer predict):
Concatenate historical exog (full length used during fit) with exog_future, then call predict with start/end covering the forecast range. This is more cumbersome—get_forecast is simpler.

Extra tips:

Ensure your time index has a frequency: df_h = df_h.asfreq('H') after resampling.

Exogenous column order and dtype must match training.

If you switch to daily data, adjust freq='D' and seasonal period (e.g., 7).

For multi-step probabilistic intervals: fc_res.conf_int().

Next Steps

Tune SARIMAX via AIC grid or pmdarima auto-ARIMA (with caution).

Try gradient boosting (XGBoost/LightGBM) on lagged features.

Add prophet or NeuralForecast/DL (LSTM/GRU) for sequences.

Perform feature importance on ML model and residual diagnostics on SARIMAX.

Build a simple inference script to update forecasts daily/hourly.
