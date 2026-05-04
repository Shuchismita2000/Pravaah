import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
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
# MAIN FORECAST FUNCTION (FIXED)
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

    # ── TREND FIT (FIXED TIME INDEX) ───────────────────────────────
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
    t_val = np.arange(len(train), len(train) + val_hours).reshape(-1, 1)
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
    t_future = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
    fc_trend = lr.predict(t_future)

    fc_health = fc_trend + repair_probs * avg_magnitude
    fc_health = np.clip(fc_health, 0.35, 0.99)

    sigma = val_score["RMSE"]
    repair_sigma = repair_probs * avg_magnitude
    total_sigma = np.sqrt(sigma**2 + repair_sigma**2)

    fc_lower = np.maximum(0.35, fc_health - 1.645 * total_sigma)
    fc_upper = np.minimum(0.99, fc_health + 1.645 * total_sigma)

    forecast_df = pd.DataFrame({
        "plant_id": plant_id,
        "timestamp": pd.date_range(series.index[-1] + pd.Timedelta(hours=1),
                                   periods=horizon, freq="h"),
        "health_forecast": np.round(fc_health, 4),
        "lower_90": np.round(fc_lower, 4),
        "upper_90": np.round(fc_upper, 4),
        "repair_probability": np.round(repair_probs, 4),
    })

    # ✅ RETURN MODEL (FIXED)
    return {
        "plant_id": plant_id,
        "model": lr,
        "slope": slope,
        "last_health": current_level,
        "last_timestamp": series.index[-1],
        "forecast_df": forecast_df,
        "repair_events": repair_events,
    }


# ══════════════════════════════════════════════════════════════════════
# WORKER (FIXED)
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

        # ✅ SAVE MODEL (NOW WORKS)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        path = MODEL_DIR / f"{plant_id}_health_model_{ts}.joblib"

        joblib.dump({
            "model": result["model"],
            "slope": result["slope"],
            "last_health": result["last_health"],
            "last_timestamp": result["last_timestamp"],
            "plant_id": plant_id,
            "horizon": horizon,
            "bounds": (0.35, 0.99),
        }, path, compress=3)

        print(f"    💾 Saved model → {path.name}")

        return {
            "status": "ok",
            "plant_id": plant_id,
            "forecast_df": result["forecast_df"],
        }

    except Exception as e:
        return {"status": "error", "plant_id": plant_id, "error": str(e)}


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
        print("❌ No forecasts generated")
        return pd.DataFrame()

    out = pd.concat(forecasts, ignore_index=True)
    out.to_csv(OUT_DIR / "health_forecasts.csv", index=False)

    print(f"\n✅ Saved forecasts → {OUT_DIR}/health_forecasts.csv")
    return out