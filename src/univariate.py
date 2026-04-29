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

from joblib import Parallel, delayed
import os
import traceback
from pathlib import Path

OUT_DIR = Path("data/forecasts")
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
    horizon    : forecast hours (default 72)
    val_hours  : validation window size (default 168 = 7 days)

    Returns
    -------
    dict with keys:
        plant_id, best_model, diagnostics, all_scores,
        forecast_72h, forecast_timestamps, forecast_df   ← always present
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

    # ── Pick winner on MAE ────────────────────────────────────────────
    best_name = min(results, key=lambda m: results[m]["scores"]["MAE"])
    print(f"\n  Winner for {plant_id}: {best_name.upper()}")
    for m, r in results.items():
        flag = " <--" if m == best_name else ""
        print(f"    {m:8s}: MAE={r['scores']['MAE']:.3f}  "
              f"RMSE={r['scores']['RMSE']:.3f}  "
              f"MAPE={r['scores']['MAPE']:.1f}%{flag}")

    # ── Refit winner on FULL series ───────────────────────────────────
    print(f"\n  Refitting {best_name} on full series for {horizon}h forecast...")

    forecast_timestamps = pd.date_range(
        start=series.index[-1] + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
    )

    fc_mw    = np.zeros(horizon)
    fc_lower = np.zeros(horizon)
    fc_upper = np.zeros(horizon)

    if best_name == "sarima":
        fit_obj, fc_mw = fit_sarima(series, horizon=horizon)
        try:
            ci       = fit_obj.get_forecast(steps=horizon).conf_int(alpha=0.10)
            fc_lower = np.maximum(0, ci.iloc[:, 0].values)
            fc_upper = ci.iloc[:, 1].values
        except Exception:
            fc_lower = np.maximum(0, fc_mw * 0.85)
            fc_upper = fc_mw * 1.15

    elif best_name == "prophet":
        _, fc_df_full = fit_prophet(series, horizon=horizon)
        fc_mw    = fc_df_full["yhat"].values
        fc_lower = fc_df_full["yhat_lower"].clip(lower=0).values
        fc_upper = fc_df_full["yhat_upper"].values

    elif best_name == "lstm":
        _, _, fc_mw = fit_lstm(series, horizon=horizon)
        rmse     = results["lstm"]["scores"]["RMSE"]
        fc_lower = np.maximum(0, fc_mw - 1.5 * rmse)
        fc_upper = fc_mw + 1.5 * rmse

    # ── Always build forecast_df ──────────────────────────────────────
    forecast_df = pd.DataFrame({
        "plant_id":    plant_id,
        "timestamp":   forecast_timestamps,
        "forecast_mw": np.round(fc_mw,    3),
        "lower_90":    np.round(fc_lower, 3),
        "upper_90":    np.round(fc_upper, 3),
        "model":       best_name,
    })

    return {
        "plant_id":            plant_id,
        "best_model":          best_name,
        "diagnostics":         diag,
        "all_scores":          {m: r["scores"] for m, r in results.items()},
        "forecast_72h":        fc_mw,               # raw array — kept for compatibility
        "forecast_timestamps": forecast_timestamps,
        "forecast_df":         forecast_df,          # ← now always built
    }





# ══════════════════════════════════════════════════════════════════════════
# WORKER — runs in its own process, one plant at a time
# Must be a top-level function (not a lambda/nested) for joblib to pickle it
# ══════════════════════════════════════════════════════════════════════════

def _process_one_plant(plant_id: str, plant_df: pd.DataFrame, horizon: int) -> dict:
    """
    Isolated worker function called by joblib for each plant.

    Returns a result dict with keys:
        status       : "ok" | "skip" | "error"
        plant_id     : str
        forecast_df  : pd.DataFrame | None
        log_row      : dict | None
        error        : str | None

    Why return a dict instead of raising:
        joblib catches worker exceptions and re-raises them in the main process,
        which breaks the other workers. Returning a structured result lets the
        main process collect all outcomes and decide what to do.
    """
    # Each worker process needs its own warning filter
    warnings.filterwarnings("ignore")

    try:
        # ── Build series ──────────────────────────────────────────────
        series = (
            plant_df
            .sort_values("timestamp")
            .set_index("timestamp")["generation"]
            .asfreq("h")
            .ffill(limit=3)
            .fillna(0)
        )

        if len(series) < 500:
            return {
                "status":      "skip",
                "plant_id":    plant_id,
                "forecast_df": None,
                "log_row":     None,
                "error":       f"insufficient_data: {len(series)} hrs",
            }

        # ── Model selection ───────────────────────────────────────────
        result = select_best_univariate_model(series, plant_id, horizon=horizon)

        fc_df = result.get("forecast_df")
        if fc_df is None or len(fc_df) == 0:
            return {
                "status":      "error",
                "plant_id":    plant_id,
                "forecast_df": None,
                "log_row":     None,
                "error":       "empty forecast_df returned",
            }

        log_row = {
            "plant_id":   plant_id,
            "n_features": len(plant_df.columns),
            "n_obs":      result["diagnostics"]["n_obs"],
            "zero_rate":  result["diagnostics"]["zero_rate"],
            "cv":         result["diagnostics"]["cv"],
            "best_model": result["best_model"],
        }
        for model_name, sc in result["all_scores"].items():
            log_row[f"{model_name}_MAE"]  = sc["MAE"]
            log_row[f"{model_name}_RMSE"] = sc["RMSE"]
            log_row[f"{model_name}_MAPE"] = sc["MAPE"]

        return {
            "status":      "ok",
            "plant_id":    plant_id,
            "forecast_df": fc_df,
            "log_row":     log_row,
            "error":       None,
        }

    except Exception as e:
        return {
            "status":      "error",
            "plant_id":    plant_id,
            "forecast_df": None,
            "log_row":     None,
            "error":       f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


# ══════════════════════════════════════════════════════════════════════════
# FLEET RUNNER — parallelised with joblib
# ══════════════════════════════════════════════════════════════════════════

def run_univariate_fleet(
    fleet_df: pd.DataFrame,
    horizon: int = 72,
    output_path: str = "data/forecasts/univariate_forecasts.csv",
    n_jobs: int = -1,
    backend: str = "loky",
) -> pd.DataFrame:
    """
    Parallelised univariate model selection and forecasting for all plants.

    Parameters
    ----------
    fleet_df    : Pre-processed feature-engineered dataframe, all plants.
                  Required columns: plant_id, timestamp, generation.

    horizon     : Forecast horizon in hours. Default 72.

    output_path : Where to save the combined forecast CSV.

    n_jobs      : Number of parallel workers.
                  -1  = use all CPU cores          (fastest, high RAM)
                   1  = sequential, no parallelism  (easiest to debug)
                   4  = use 4 cores                 (balanced)

                  Rule of thumb for this workload:
                    - SARIMA + Prophet are CPU-bound → more cores = faster
                    - LSTM is GPU-bound if CUDA available → set n_jobs=1
                      and let LSTM use the full GPU per plant sequentially
                    - If no GPU: n_jobs = min(cpu_count, n_plants // 2)

    backend     : joblib backend.
                  "loky"      = default, process-based, safest for statsmodels/prophet
                  "threading" = thread-based, use only if models release the GIL
                                (they mostly don't — stick with loky)

    Returns
    -------
    pd.DataFrame : Combined forecast for all plants (long format).
    """
    # ── Validate ──────────────────────────────────────────────────────
    required = {"plant_id", "timestamp", "generation"}
    missing  = required - set(fleet_df.columns)
    if missing:
        raise ValueError(
            f"fleet_df missing required columns: {missing}\n"
            f"Run preprocess() first — it aliases actual_generation_mw → generation."
        )

    fleet_df = fleet_df.copy()
    fleet_df["timestamp"] = pd.to_datetime(fleet_df["timestamp"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Split into per-plant dataframes ───────────────────────────────
    # Do this BEFORE joblib so we're passing DataFrames, not a GroupBy object
    # (GroupBy objects are not picklable across processes)
    plant_groups = [
        (plant_id, plant_df.reset_index(drop=True))
        for plant_id, plant_df in fleet_df.groupby("plant_id")
    ]
    n_plants = len(plant_groups)

    # ── Resolve n_jobs ────────────────────────────────────────────────
    import multiprocessing
    max_cores   = multiprocessing.cpu_count()
    resolved_jobs = max_cores if n_jobs == -1 else min(n_jobs, max_cores)
    # Never run more workers than plants — wastes overhead
    resolved_jobs = min(resolved_jobs, n_plants)

    print(f"\n{'═'*60}")
    print(f"  FLEET UNIVARIATE FORECAST")
    print(f"  Plants   : {n_plants}")
    print(f"  Horizon  : {horizon}h")
    print(f"  Workers  : {resolved_jobs} / {max_cores} cores  (n_jobs={n_jobs})")
    print(f"  Backend  : {backend}")
    print(f"{'═'*60}\n")

    # ── Parallel execution ────────────────────────────────────────────
    # verbose=10 prints a progress line per completed job
    results = Parallel(n_jobs=resolved_jobs, backend=backend, verbose=10)(
        delayed(_process_one_plant)(plant_id, plant_df, horizon)
        for plant_id, plant_df in plant_groups
    )

    # ── Collect results ───────────────────────────────────────────────
    all_forecasts  = []
    selection_log  = []
    failed_plants  = []

    for res in results:
        pid = res["plant_id"]
        if res["status"] == "ok":
            all_forecasts.append(res["forecast_df"])
            selection_log.append(res["log_row"])
            print(f"  ✓ {pid} — {res['log_row']['best_model']}"
                  f"  MAE={res['log_row'].get(res['log_row']['best_model']+'_MAE','?')}")
        elif res["status"] == "skip":
            print(f"  ⊘ {pid} — skipped: {res['error']}")
            failed_plants.append({"plant_id": pid, "error": res["error"]})
        else:
            print(f"  ✗ {pid} — ERROR: {res['error'].splitlines()[0]}")
            failed_plants.append({"plant_id": pid, "error": res["error"]})

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  FLEET COMPLETE")
    print(f"  Successful : {len(all_forecasts)} / {n_plants} plants")
    print(f"  Failed     : {len(failed_plants)} plants")

    log_df = pd.DataFrame(selection_log)
    if len(log_df):
        print(f"\n  Model distribution:")
        for model, cnt in log_df["best_model"].value_counts().items():
            print(f"    {model:10s}: {cnt:3d} plants ({cnt/len(log_df):.0%})")
    print(f"{'═'*60}\n")

    # ── Save logs ─────────────────────────────────────────────────────
    if len(log_df):
        log_df.to_csv(OUT_DIR / "model_selection_log.csv", index=False)

    if failed_plants:
        pd.DataFrame(failed_plants).to_csv(OUT_DIR / "failed_plants.csv", index=False)
        print(f"  Failed details saved → {OUT_DIR}/failed_plants.csv")

    # ── Guard concat ──────────────────────────────────────────────────
    if not all_forecasts:
        print("\n[FATAL] No forecasts generated.")
        print("  → Set n_jobs=1 to see full error output per plant")
        print("  → Check failed_plants.csv for error details")
        return pd.DataFrame()

    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    forecast_df.to_csv(output_path, index=False)
    print(f"  Forecast saved → {output_path}")
    return forecast_df
