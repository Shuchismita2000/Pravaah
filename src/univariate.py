"""
Step A — Univariate generation forecasting
Purpose: produce 72-hour generation forecasts per plant
         so those values can feed into the multivariate pipeline as lagged features.

Models compared per plant:
  - SARIMA   (captures seasonality, no external regressors)
  - Prophet  (handles multiple seasonalities + holidays cleanly)
  - LSTM     (learns non-linear temporal patterns)

Output: best_model per plant + 72-hour forecast array
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TIME SERIES DIAGNOSTICS
# Run this first per plant to understand the series before choosing a model
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_series(series: pd.Series, plant_id: str) -> dict:
    """
    Stationarity, seasonality strength, and missing rate.
    Tells you which model family is most appropriate.

    Returns a dict of diagnostics used to auto-select model.
    """
    series = series.dropna()

    # 1. Stationarity — ADF test
    adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
    is_stationary = adf_p < 0.05

    # 2. Seasonal strength — variance ratio at lag 24 (daily) and lag 168 (weekly)
    acf_vals = acf(series, nlags=170, fft=True)
    daily_strength  = abs(acf_vals[24])
    weekly_strength = abs(acf_vals[168])

    # 3. Zero rate (hours with no generation — nights for solar)
    zero_rate = (series == 0).mean()

    # 4. Coefficient of variation (how volatile)
    cv = series.std() / (series.mean() + 1e-6)

    diag = {
        "plant_id":        plant_id,
        "n_obs":           len(series),
        "is_stationary":   is_stationary,
        "adf_pvalue":      round(adf_p, 4),
        "daily_acf":       round(daily_strength, 3),
        "weekly_acf":      round(weekly_strength, 3),
        "zero_rate":       round(zero_rate, 3),
        "cv":              round(cv, 3),
        "mean_mw":         round(series.mean(), 2),
        "recommended_model": _recommend_model(is_stationary, daily_strength,
                                               weekly_strength, zero_rate, cv),
    }
    print(f"[{plant_id}] stationary={is_stationary} | daily_acf={daily_strength:.2f} "
          f"| zero_rate={zero_rate:.1%} | CV={cv:.2f} → {diag['recommended_model']}")
    return diag


def _recommend_model(stationary, daily, weekly, zero_rate, cv):
    # High zero rate = solar plant with clear day/night pattern → Prophet handles this best
    if zero_rate > 0.35:
        return "prophet"
    # Strong weekly seasonality + non-stationary → SARIMA with seasonal differencing
    if weekly > 0.3 and not stationary:
        return "sarima"
    # High volatility or complex non-linear pattern → LSTM
    if cv > 1.2:
        return "lstm"
    # Default: Prophet (robust, handles missing, multiple seasonalities)
    return "prophet"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TRAIN / VALIDATION SPLIT
# Always split on time, never shuffle for time series
# ══════════════════════════════════════════════════════════════════════════════

def time_split(series: pd.Series, val_hours: int = 168):
    """
    Last `val_hours` hours for validation (default = 7 days).
    Everything before = training.
    """
    train = series.iloc[:-val_hours]
    val   = series.iloc[-val_hours:]
    return train, val


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # MAPE — skip zeros (common for solar at night)
    mask = actual > 0.1
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.sum() else np.nan
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE": round(mape, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SARIMA
# Good for: wind plants, stationary or near-stationary series
# ══════════════════════════════════════════════════════════════════════════════

def fit_sarima(train: pd.Series, horizon: int = 72) -> tuple:
    """
    SARIMA(1,1,1)(1,1,1,24) — daily seasonality.
    Auto-selects d based on stationarity test.

    Returns: (model_fit, forecast_array)
    """
    # Determine differencing order
    _, p_val, *_ = adfuller(train.dropna())
    d = 0 if p_val < 0.05 else 1

    model = SARIMAX(
        train,
        order=(1, d, 1),
        seasonal_order=(1, 1, 1, 24),  # 24-hour daily seasonality
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    forecast = fit.forecast(steps=horizon)
    forecast = np.maximum(0, forecast)  # generation can't be negative

    print(f"  SARIMA AIC={fit.aic:.1f} | BIC={fit.bic:.1f}")
    return fit, forecast.values


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PROPHET
# Good for: solar plants, multiple seasonalities, missing data, zero-heavy series
# ══════════════════════════════════════════════════════════════════════════════

KARNATAKA_HOLIDAYS = pd.DataFrame({
    "holiday": ["Republic Day", "Independence Day", "Gandhi Jayanti", "Karnataka Rajyotsava"] * 3,
    "ds": pd.to_datetime([
        "2023-01-26", "2023-08-15", "2023-10-02", "2023-11-01",
        "2024-01-26", "2024-08-15", "2024-10-02", "2024-11-01",
        "2025-01-26", "2025-08-15", "2025-10-02", "2025-11-01",
    ]),
})


def fit_prophet(train: pd.Series, horizon: int = 72) -> tuple:
    """
    Prophet with daily + weekly + yearly seasonality and Karnataka holidays.
    Handles the zero-at-night pattern for solar naturally.

    Returns: (model, forecast_df with 'yhat' column)
    """
    df = train.reset_index()
    df.columns = ["ds", "y"]

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        holidays=KARNATAKA_HOLIDAYS,
        seasonality_mode="multiplicative",  # better for solar (scales with capacity)
        changepoint_prior_scale=0.05,        # conservative — don't overfit recent changes
        interval_width=0.90,
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=horizon, freq="h")
    forecast = model.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)

    future_only = forecast.tail(horizon)
    print(f"  Prophet fitted — {horizon}h forecast range: "
          f"[{future_only['yhat'].min():.1f}, {future_only['yhat'].max():.1f}] MW")
    return model, future_only


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LSTM
# Good for: hybrid plants, high volatility, complex non-linear patterns
# ══════════════════════════════════════════════════════════════════════════════

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=72):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # take last timestep output


def fit_lstm(
    train: pd.Series,
    horizon: int = 72,
    lookback: int = 168,   # 7 days of hourly history as input window
    epochs: int = 50,
    batch_size: int = 32,
) -> tuple:
    """
    LSTM trained on sliding windows of `lookback` hours to predict next `horizon` hours.

    Returns: (model, scaler, forecast_array)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Scale to [0, 1]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()

    # Build supervised dataset: X = window of lookback, y = next horizon
    X_list, y_list = [], []
    for i in range(len(scaled) - lookback - horizon + 1):
        X_list.append(scaled[i : i + lookback])
        y_list.append(scaled[i + lookback : i + lookback + horizon])

    X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1).to(device)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).to(device)

    model = LSTMForecaster(output_size=horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X, y)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  LSTM epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.5f}")

    # Forecast: use last `lookback` hours of training data as seed
    model.eval()
    with torch.no_grad():
        seed = torch.tensor(scaled[-lookback:], dtype=torch.float32)
        seed = seed.unsqueeze(0).unsqueeze(-1).to(device)
        pred_scaled = model(seed).cpu().numpy().flatten()

    forecast = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    forecast = np.maximum(0, forecast)
    return model, scaler, forecast


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MODEL SELECTION: run all, pick best on validation MAE
# ══════════════════════════════════════════════════════════════════════════════

def select_best_univariate_model(
    series: pd.Series,
    plant_id: str,
    plant_type: str,
    horizon: int = 72,
    val_hours: int = 168,
) -> dict:
    """
    Fits SARIMA, Prophet, and LSTM on train split.
    Evaluates each on val split.
    Returns the winner + its 72-hour forecast from the full series.

    Parameters
    ----------
    series     : hourly generation series with DatetimeIndex
    plant_id   : for logging
    plant_type : "Solar" | "Wind" | "Hybrid"
    horizon    : forecast hours (default 72)
    val_hours  : validation window size (default 168 = 7 days)

    Returns
    -------
    dict with keys: plant_id, best_model, scores, forecast_72h, forecast_timestamps
    """
    diag = diagnose_series(series, plant_id)
    train, val = time_split(series, val_hours)
    results = {}

    # ── SARIMA ──
    try:
        _, sarima_fc = fit_sarima(train, horizon=val_hours)
        results["sarima"] = {
            "scores": evaluate(val.values, sarima_fc[:val_hours]),
            "model_tag": "sarima",
        }
    except Exception as e:
        print(f"  [SARIMA failed] {e}")

    # ── Prophet ──
    try:
        _, prophet_fc = fit_prophet(train, horizon=val_hours)
        results["prophet"] = {
            "scores": evaluate(val.values, prophet_fc["yhat"].values),
            "model_tag": "prophet",
        }
    except Exception as e:
        print(f"  [Prophet failed] {e}")

    # ── LSTM ──
    try:
        _, _, lstm_fc = fit_lstm(train, horizon=val_hours)
        results["lstm"] = {
            "scores": evaluate(val.values, lstm_fc[:val_hours]),
            "model_tag": "lstm",
        }
    except Exception as e:
        print(f"  [LSTM failed] {e}")

    if not results:
        raise RuntimeError(f"All models failed for {plant_id}")

    # Pick winner on MAE
    best_name = min(results, key=lambda m: results[m]["scores"]["MAE"])
    print(f"\n  Winner for {plant_id}: {best_name.upper()}")
    for m, r in results.items():
        flag = " <--" if m == best_name else ""
        print(f"    {m:8s}: MAE={r['scores']['MAE']:.3f}  "
              f"RMSE={r['scores']['RMSE']:.3f}  "
              f"MAPE={r['scores']['MAPE']:.1f}%{flag}")

    # Refit winner on FULL series, produce real 72h forecast
    print(f"\n  Refitting {best_name} on full series for 72h forecast...")
    forecast_timestamps = pd.date_range(
        start=series.index[-1] + pd.Timedelta(hours=1),
        periods=horizon, freq="h"
    )

    if best_name == "sarima":
        _, fc = fit_sarima(series, horizon=horizon)
    elif best_name == "prophet":
        _, fc_df = fit_prophet(series, horizon=horizon)
        fc = fc_df["yhat"].values
    else:
        _, _, fc = fit_lstm(series, horizon=horizon)

    return {
        "plant_id":            plant_id,
        "plant_type":          plant_type,
        "best_model":          best_name,
        "diagnostics":         diag,
        "validation_scores":   {m: r["scores"] for m, r in results.items()},
        "forecast_72h":        fc,
        "forecast_timestamps": forecast_timestamps,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FLEET RUNNER
# Run univariate selection for all 50 plants, save results
# ══════════════════════════════════════════════════════════════════════════════

def run_univariate_fleet(
    generation_df: pd.DataFrame,
    plant_master_df: pd.DataFrame,
    horizon: int = 72,
    output_path: str = "data/univariate_forecasts.parquet",
) -> pd.DataFrame:
    """
    Runs select_best_univariate_model() for every plant.
    Saves a long-format parquet with columns:
        plant_id | timestamp | forecast_mw | model_used

    This output is the input to your multivariate FE pipeline as
    the 'generation' lag/rolling features for the forecast window.

    Usage
    -----
    gen = pd.read_csv("data/generation_raw.csv", parse_dates=["timestamp"])
    pm  = pd.read_csv("data/plant_master.csv")
    forecast_df = run_univariate_fleet(gen, pm)
    """
    gen_df = generation_df.copy()
    gen_df["timestamp"] = pd.to_datetime(gen_df["timestamp"])

    all_forecasts = []
    model_selection_log = []

    for _, plant in plant_master_df.iterrows():
        pid   = plant["plant_id"]
        ptype = plant["plant_type"]
        print(f"\n{'='*55}")
        print(f"Plant: {pid} — {plant['plant_name']} ({ptype})")
        print("="*55)

        plant_gen = (
            gen_df[gen_df["plant_id"] == pid]
            .set_index("timestamp")["actual_generation_mw"]
            .sort_index()
            .asfreq("h")           # enforce hourly frequency
            .fillna(method="ffill", limit=3)
            .fillna(0)
        )

        if len(plant_gen) < 500:
            print(f"  [SKIP] insufficient data ({len(plant_gen)} hrs)")
            continue

        try:
            result = select_best_univariate_model(
                plant_gen, pid, ptype, horizon=horizon
            )
            # Long format forecast rows
            for ts, mw in zip(result["forecast_timestamps"], result["forecast_72h"]):
                all_forecasts.append({
                    "plant_id":    pid,
                    "timestamp":   ts,
                    "forecast_mw": round(float(mw), 3),
                    "model_used":  result["best_model"],
                })
            model_selection_log.append({
                "plant_id":   pid,
                "plant_type": ptype,
                "best_model": result["best_model"],
                **{f"{m}_MAE": v["MAE"] for m, v in result["validation_scores"].items()},
            })

        except Exception as e:
            print(f"  [ERROR] {pid}: {e}")

    forecast_df = pd.DataFrame(all_forecasts)
    forecast_df.to_parquet(output_path, index=False)

    log_df = pd.DataFrame(model_selection_log)
    log_df.to_csv("data/model_selection_log.csv", index=False)

    print(f"\n{'='*55}")
    print(f"Fleet univariate forecast complete")
    print(f"  Plants processed : {log_df.shape[0]}")
    print(f"  Model distribution:\n{log_df['best_model'].value_counts().to_string()}")
    print(f"  Saved → {output_path}")
    return forecast_df

    