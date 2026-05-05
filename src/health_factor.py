import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from datetime import datetime

warnings.filterwarnings("ignore")

OUT_DIR   = Path("data/forecasts")
MODEL_DIR = Path("models/health_factor")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════

def _time_split(series: pd.Series, val_hours: int = 168):
    train = series.iloc[:-val_hours]
    val   = series.iloc[-val_hours:]
    return train, val


def _evaluate(actual: np.ndarray, predicted: np.ndarray, name: str) -> dict:
    n    = min(len(actual), len(predicted))
    act  = actual[:n]
    pred = np.maximum(0, predicted[:n])

    mae  = mean_absolute_error(act, pred)
    rmse = np.sqrt(mean_squared_error(act, pred))

    nonzero = act > 0.01
    mape = (
        float(np.mean(np.abs((act[nonzero] - pred[nonzero]) / (act[nonzero] + 1e-6))) * 100)
        if nonzero.sum() > 5 else np.nan
    )

    print(f"    {name:20s} → MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.1f}%")
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2)}


# ══════════════════════════════════════════════════════════════════════
# REPAIR DETECTION
# ══════════════════════════════════════════════════════════════════════

def _detect_repair_events(series: pd.Series) -> pd.DataFrame:
    diff   = series.diff()
    repair = diff[diff > 0.05]

    events = []
    for ts, magnitude in repair.items():
        events.append({
            "timestamp": ts,
            "health_before": float(series[ts] - magnitude),
            "health_after": float(series[ts]),
            "magnitude": round(float(magnitude), 4),
        })

    return pd.DataFrame(events) if events else pd.DataFrame(
        columns=["timestamp", "health_before", "health_after", "magnitude"]
    )


def _estimate_repair_probability(series, repair_events, horizon):
    current_health = float(series.iloc[-1])
    n_hours = len(series)

    base_rate = len(repair_events) / n_hours if n_hours > 0 else 0.001

    if current_health < 0.55:
        urgency = 3.0
    elif current_health < 0.65:
        urgency = 2.0
    elif current_health < 0.75:
        urgency = 1.0
    else:
        urgency = 0.3

    if len(repair_events) > 0:
        last_repair = pd.Timestamp(repair_events["timestamp"].max())
        hours_since = (series.index[-1] - last_repair).total_seconds() / 3600
    else:
        hours_since = n_hours

    time_factor = min(2.0, hours_since / (30 * 24))
    hourly_prob = min(0.05, base_rate * urgency * time_factor)

    return np.full(horizon, hourly_prob)


# ══════════════════════════════════════════════════════════════════════
# MAIN FORECAST FUNCTION
# ══════════════════════════════════════════════════════════════════════

def forecast_health_factor_one_plant(
    plant_id: str,
    series: pd.Series,
    lifecycle_df: pd.DataFrame,
    horizon: int = 72,
    val_hours: int = 168,
):
    print(f"\n[{plant_id}] Health forecast ({len(series):,} obs)")

    series = series.clip(0.35, 0.99)
    train, val = _time_split(series, val_hours)

    # ── TREND FIT ──────────────────────────────────────────────────
    window = min(len(train), 30 * 24)
    recent = train.iloc[-window:]

    t_full   = np.arange(len(series))
    t_recent = t_full[-window:].reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(t_recent, recent.values)

    slope         = float(lr.coef_[0])
    current_level = float(series.iloc[-1])

    print(f"    Current health: {current_level:.4f}")
    print(f"    Monthly slope: {slope * 24 * 30:.4f}")

    # ── VALIDATION ────────────────────────────────────────────────
    t_val    = np.arange(len(train), len(train) + val_hours).reshape(-1, 1)
    val_pred = lr.predict(t_val)
    val_pred = np.clip(val_pred, 0.35, 0.99)
    val_score = _evaluate(val.values, val_pred, "linear trend")

    # ── REPAIR LOGIC ──────────────────────────────────────────────
    repair_events = _detect_repair_events(series)
    repair_probs  = _estimate_repair_probability(series, repair_events, horizon)
    avg_magnitude = (
        repair_events["magnitude"].mean() if len(repair_events) > 0 else 0.20
    )

    # ── FORECAST ──────────────────────────────────────────────────
    t_future  = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
    fc_trend  = lr.predict(t_future)
    fc_health = fc_trend + repair_probs * avg_magnitude
    fc_health = np.clip(fc_health, 0.35, 0.99)

    sigma        = val_score["RMSE"]
    repair_sigma = repair_probs * avg_magnitude
    total_sigma  = np.sqrt(sigma**2 + repair_sigma**2)

    fc_lower = np.maximum(0.35, fc_health - 1.645 * total_sigma)
    fc_upper = np.minimum(0.99, fc_health + 1.645 * total_sigma)

    forecast_df = pd.DataFrame({
        "plant_id":           plant_id,
        "timestamp":          pd.date_range(
                                  series.index[-1] + pd.Timedelta(hours=1),
                                  periods=horizon, freq="h"),
        "health_forecast":    np.round(fc_health, 4),
        "lower_90":           np.round(fc_lower,  4),
        "upper_90":           np.round(fc_upper,  4),
        "repair_probability": np.round(repair_probs, 4),
    })

    return {
        "plant_id":       plant_id,
        "model":          lr,               # sklearn LinearRegression — picklable
        "slope":          slope,
        "val_rmse":       val_score["RMSE"],
        "last_health":    current_level,
        "last_t_index":   int(t_full[-1]),  # needed to continue the time axis at predict time
        "last_timestamp": series.index[-1],
        "avg_magnitude":  avg_magnitude,
        "repair_events":  repair_events,
        "forecast_df":    forecast_df,
    }


# ══════════════════════════════════════════════════════════════════════
# WORKER
# ══════════════════════════════════════════════════════════════════════

def _health_worker(plant_id, plant_df, lifecycle_df, horizon):
    try:
        series = (
            plant_df.sort_values("timestamp")
            .set_index("timestamp")["health_factor"]
            .asfreq("h").ffill(limit=24).fillna(0.85)
        )

        if len(series) < 500:
            return {"status": "skip", "plant_id": plant_id}

        result = forecast_health_factor_one_plant(
            plant_id, series, lifecycle_df, horizon
        )

        # ── Save pkl ──────────────────────────────────────────────
        ts         = datetime.now().strftime("%Y%m%d_%H%M")
        model_path = MODEL_DIR / f"{plant_id}_health_{ts}.pkl"

        payload = {
            "model_name":     "linear_trend",
            "model":          result["model"],          # sklearn LR — call model.predict(t_future)
            "plant_id":       plant_id,
            "horizon":        horizon,
            "slope":          result["slope"],
            "val_rmse":       result["val_rmse"],
            "last_health":    result["last_health"],
            "last_t_index":   result["last_t_index"],   # continue time axis from here
            "last_timestamp": result["last_timestamp"],
            "avg_magnitude":  result["avg_magnitude"],
            "bounds":         (0.35, 0.99),
        }

        with open(model_path, "wb") as f:
            pickle.dump(payload, f)

        print(f"    Saved pkl → {model_path.name}")

        return {
            "status":      "ok",
            "plant_id":    plant_id,
            "forecast_df": result["forecast_df"],
        }

    except Exception as e:
        return {"status": "error", "plant_id": plant_id, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════
# PREDICT FROM SAVED PKL
# ══════════════════════════════════════════════════════════════════════

def predict_health_factor(
    pkl_path: str,
    timestamps,         # pd.DatetimeIndex | pd.Series | list of datetime-like
    horizon: int = None,
) -> pd.DataFrame:
    """
    Load a saved health pkl and predict for the given timestamps.

    Parameters
    ----------
    pkl_path   : path to .pkl saved by _health_worker
    timestamps : exact timestamps you want predictions for.
                 Accepts pd.DatetimeIndex, pd.Series, or a list of datetime-like.
                 If None, falls back to building `horizon` hours from last_timestamp.
    horizon    : only used when timestamps=None.

    Returns
    -------
    pd.DataFrame with columns:
        plant_id, timestamp, health_forecast, lower_90, upper_90, repair_probability, method

    Example
    -------
    ts = pd.date_range("2026-06-01", periods=72, freq="h")
    df = predict_health_factor("models/health_factor/PLANT_001_health_20260505_1400.pkl", ts)

    # or from a DataFrame column:
    df = predict_health_factor(pkl_path, timestamps=my_df["timestamp"])
    """
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    model          = payload["model"]
    plant_id       = payload["plant_id"]
    last_t_index   = payload["last_t_index"]    # integer time index at training cutoff
    last_timestamp = payload["last_timestamp"]  # pd.Timestamp at training cutoff
    last_health    = payload["last_health"]
    avg_magnitude  = payload["avg_magnitude"]
    val_rmse       = payload["val_rmse"]
    lo, hi         = payload["bounds"]

    # ── Resolve timestamps ────────────────────────────────────────
    if timestamps is not None:
        if isinstance(timestamps, pd.Series):
            forecast_timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps.values))
        elif isinstance(timestamps, pd.DatetimeIndex):
            forecast_timestamps = timestamps
        else:
            forecast_timestamps = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    else:
        h = horizon or payload["horizon"]
        forecast_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=h,
            freq="h",
        )

    n = len(forecast_timestamps)

    # ── Rebuild integer time axis (same scale as training) ────────
    # Each forecast timestamp maps to last_t_index + hours_offset
    hours_offset = np.array([
        (ts - last_timestamp).total_seconds() / 3600
        for ts in forecast_timestamps
    ], dtype=float)
    t_future = (last_t_index + hours_offset).reshape(-1, 1)

    # ── Trend forecast ────────────────────────────────────────────
    fc_trend = model.predict(t_future)

    # ── Repair probability (constant urgency based on last health) ─
    if last_health < 0.55:
        urgency = 3.0
    elif last_health < 0.65:
        urgency = 2.0
    elif last_health < 0.75:
        urgency = 1.0
    else:
        urgency = 0.3

    hourly_prob  = min(0.05, 0.001 * urgency)
    repair_probs = np.full(n, hourly_prob)

    # ── Combine ───────────────────────────────────────────────────
    fc_health = fc_trend + repair_probs * avg_magnitude
    fc_health = np.clip(fc_health, lo, hi)

    repair_sigma = repair_probs * avg_magnitude
    total_sigma  = np.sqrt(val_rmse**2 + repair_sigma**2)

    fc_lower = np.maximum(lo, fc_health - 1.645 * total_sigma).copy()
    fc_upper = np.minimum(hi, fc_health + 1.645 * total_sigma).copy()

    return pd.DataFrame({
        "plant_id":           plant_id,
        "timestamp":          forecast_timestamps,
        "health_forecast":    np.round(fc_health,    4),
        "lower_90":           np.round(fc_lower,     4),
        "upper_90":           np.round(fc_upper,     4),
        "repair_probability": np.round(repair_probs, 4),
        "method":             "linear_trend",
    })


# ══════════════════════════════════════════════════════════════════════
# FLEET RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_health_fleet(merged_df, lifecycle_df, horizon=72, n_jobs=-1):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = merged_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    groups = [(pid, g.copy()) for pid, g in df.groupby("plant_id")]

    print(f"\nHEALTH FORECAST | Plants: {len(groups)}")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_health_worker)(pid, grp, lifecycle_df, horizon)
        for pid, grp in groups
    )

    forecasts = [r["forecast_df"] for r in results if r["status"] == "ok"]

    if not forecasts:
        print("No forecasts generated")
        return pd.DataFrame()

    out = pd.concat(forecasts, ignore_index=True)
    out.to_csv(OUT_DIR / "health_forecasts.csv", index=False)

    print(f"\nSaved forecasts → {OUT_DIR}/health_forecasts.csv")
    return out