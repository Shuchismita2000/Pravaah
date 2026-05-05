"""
availability_mw:
  - Driven by planned maintenance schedules + forced outages
  - Strong weekly cycle (maintenance often on weekends / fixed intervals)
  - Step-function behaviour: drops suddenly (outage), recovers to capacity
  - Bounded: [0, capacity_mw] — never exceeds installed capacity
  → Best models: Prophet (handles weekly + yearly cycles cleanly)
                 + capacity baseline (assume full availability)
  → DO NOT use SARIMA (non-stationarity from outage steps breaks it)
  → DO NOT use LSTM (overkill, insufficient outage events to learn from)
"""

import warnings
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("[WARN] prophet not installed — availability will use capacity baseline only")

OUT_DIR   = Path("data/forecasts")
MODEL_DIR = Path("models/availability")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
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
    print(f"    {name:25s} → MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.1f}%")
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2)}


# ══════════════════════════════════════════════════════════════════════
# PART 1 — AVAILABILITY FORECAST
# ══════════════════════════════════════════════════════════════════════

def _capacity_baseline_forecast(capacity_mw: float, n_periods: int) -> np.ndarray:
    return np.full(n_periods, capacity_mw, dtype=float)


def _prophet_availability_forecast(
    train: pd.Series,
    capacity_mw: float,
    horizon: int,
) -> tuple:
    if not HAS_PROPHET:
        raise ImportError("prophet not installed")

    df = train.reset_index()
    df.columns = ["ds", "y"]
    df["floor"] = 0.0
    df["cap"]   = float(capacity_mw)

    model = Prophet(
        growth="logistic",
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.90,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5, mode="additive")
    model.fit(df)

    future = model.make_future_dataframe(periods=horizon, freq="h")
    future["floor"] = 0.0
    future["cap"]   = float(capacity_mw)

    forecast = model.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(lower=0, upper=capacity_mw)
    return model, forecast.tail(horizon).reset_index(drop=True)


def forecast_availability_one_plant(
    plant_id: str,
    series: pd.Series,
    capacity_mw: float,
    horizon: int = 72,
    val_hours: int = 168,
) -> dict:
    print(f"\n  [{plant_id}] Availability forecast ({len(series):,} obs, "
          f"capacity={capacity_mw:.1f} MW)...")

    series = series.clip(lower=0, upper=capacity_mw)
    train, val = _time_split(series, val_hours)

    results = {}

    # ── 1. Capacity baseline ──────────────────────────────────────
    baseline_val = _capacity_baseline_forecast(capacity_mw, len(val))
    results["capacity_baseline"] = _evaluate(val.values, baseline_val, "capacity baseline")

    # ── 2. Prophet ────────────────────────────────────────────────
    prophet_model  = None
    prophet_val_fc = None
    if HAS_PROPHET:
        try:
            prophet_model, prophet_val_df = _prophet_availability_forecast(
                train, capacity_mw, horizon=val_hours
            )
            prophet_val_fc     = prophet_val_df["yhat"].values
            results["prophet"] = _evaluate(val.values, prophet_val_fc, "prophet")
        except Exception as e:
            print(f"    Prophet failed: {e}")
            prophet_model = None

    # ── Pick winner ───────────────────────────────────────────────
    best = min(results, key=lambda m: results[m]["MAE"])
    print(f"    Winner: {best} (MAE={results[best]['MAE']:.3f})")

    # ── Forecast on full series ───────────────────────────────────
    forecast_timestamps = pd.date_range(
        start=series.index[-1] + pd.Timedelta(hours=1),
        periods=horizon, freq="h",
    )

    baseline_fc = _capacity_baseline_forecast(capacity_mw, horizon)

    if best == "prophet" and HAS_PROPHET:
        # Re-fit on full series to get the final model object
        prophet_model, fc_df_full = _prophet_availability_forecast(series, capacity_mw, horizon)
        prophet_fc = fc_df_full["yhat"].values
        fc_lower   = fc_df_full["yhat_lower"].clip(lower=0).values.copy()
        fc_upper   = fc_df_full["yhat_upper"].clip(upper=capacity_mw).values.copy()
        blended_fc = 0.70 * prophet_fc + 0.30 * baseline_fc
        method     = "blend_prophet_baseline"
    else:
        prophet_model = None
        blended_fc    = baseline_fc.copy()
        rmse          = results["capacity_baseline"]["RMSE"]
        fc_lower      = np.maximum(0, blended_fc - 1.5 * rmse)
        fc_upper      = np.minimum(capacity_mw, blended_fc + 1.5 * rmse)
        method        = "capacity_baseline"

    blended_fc = np.clip(blended_fc, 0, capacity_mw)
    fc_lower   = np.maximum(0, fc_lower)
    fc_upper   = np.minimum(capacity_mw, fc_upper)

    forecast_df = pd.DataFrame({
        "plant_id":              plant_id,
        "timestamp":             forecast_timestamps,
        "availability_forecast": np.round(blended_fc, 2),
        "lower_90":              np.round(fc_lower,   2),
        "upper_90":              np.round(fc_upper,   2),
        "method":                method,
    })

    uptime_pct = (blended_fc > 0).mean() * 100
    avg_avail  = blended_fc.mean()
    print(f"    Avg forecast: {avg_avail:.1f} MW  "
          f"| Uptime: {uptime_pct:.1f}%  "
          f"| Capacity: {capacity_mw:.1f} MW")

    return {
        "plant_id":     plant_id,
        "scores":       results,
        "best_model":   best,
        "model_obj":    prophet_model,   # fitted Prophet, or None for baseline
        "val_rmse":     results[best]["RMSE"],
        "capacity_mw":  capacity_mw,
        "forecast_df":  forecast_df,
    }


# ══════════════════════════════════════════════════════════════════════
# PART 2 — WORKER + SAVE PKL
# ══════════════════════════════════════════════════════════════════════
def _availability_worker(plant_id, plant_df, horizon):
    warnings.filterwarnings("ignore")
    
    # ── Forecast (keep existing try/except) ──────────────────────
    try:
        capacity_mw = plant_df["capacity_mw"].median()
        series = (
            plant_df.sort_values("timestamp")
            .set_index("timestamp")["availability_mw"]
            .asfreq("h")
            .fillna(capacity_mw)
            .clip(lower=0, upper=capacity_mw)
        )
        if len(series) < 500:
            return {"status": "skip", "plant_id": plant_id,
                    "error": f"only {len(series)} obs"}

        result = forecast_availability_one_plant(plant_id, series, capacity_mw, horizon)

    except Exception as e:
        import traceback
        return {"status": "error", "plant_id": plant_id,
                "forecast_df": None,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}

    # ── Pkl save (separate try/except — never kills the forecast) ─
    best = result["best_model"]
    payload = {
        "model_name":  best,
        "model":       result["model_obj"] if best == "prophet" else None,
        "plant_id":    plant_id,
        "capacity_mw": float(result["capacity_mw"]),
        "horizon":     horizon,
        "val_rmse":    result["val_rmse"],
    }
    try:
        ts         = datetime.now().strftime("%Y%m%d_%H%M")
        model_path = MODEL_DIR / f"{plant_id}_availability_{ts}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"    Saved pkl → {model_path.name}")
    except Exception as e:
        print(f"    [WARN] pkl save failed for {plant_id}: {e}")
        # Forecast result is still returned — don't re-raise

    return {"status": "ok", "plant_id": plant_id,
            "forecast_df": result["forecast_df"],
            "best_model":  best}
# ══════════════════════════════════════════════════════════════════════
# PART 3 — PREDICT FROM SAVED PKL
# ══════════════════════════════════════════════════════════════════════

def predict_availability(
    pkl_path: str,
    timestamps,           # pd.DatetimeIndex | pd.Series | list of datetime-like
    horizon: int = None,  # fallback when timestamps=None
) -> pd.DataFrame:
    """
    Load a saved availability pkl and predict for the given timestamps.

    Parameters
    ----------
    pkl_path   : path to .pkl saved by _availability_worker
    timestamps : exact timestamps you want predictions for.
                 Accepts pd.DatetimeIndex, pd.Series, or list of datetime-like.
                 If None, builds `horizon` hours from now.
    horizon    : only used when timestamps=None.

    Returns
    -------
    pd.DataFrame with columns:
        plant_id, timestamp, availability_forecast, lower_90, upper_90, method

    Example
    -------
    ts = pd.date_range("2026-06-01", periods=72, freq="h")
    df = predict_availability("models/availability/PLANT_001_availability_20260505_1400.pkl", ts)

    # or from a DataFrame column:
    df = predict_availability(pkl_path, timestamps=my_df["timestamp"])
    """
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    model_name  = payload["model_name"]
    plant_id    = payload["plant_id"]
    capacity_mw = payload["capacity_mw"]
    val_rmse    = payload["val_rmse"]

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
            start=pd.Timestamp.now().floor("h") + pd.Timedelta(hours=1),
            periods=h,
            freq="h",
        )

    n = len(forecast_timestamps)

    # ── Prophet ───────────────────────────────────────────────────
    if model_name == "prophet":
        model  = payload["model"]
        future = pd.DataFrame({"ds": forecast_timestamps})
        future["floor"] = 0.0
        future["cap"]   = float(capacity_mw)
        fc_df  = model.predict(future)

        prophet_fc = fc_df["yhat"].clip(0, capacity_mw).values.copy()
        fc_lower   = fc_df["yhat_lower"].clip(lower=0).values.copy()
        fc_upper   = fc_df["yhat_upper"].clip(upper=capacity_mw).values.copy()

        # Blend with baseline (same ratio as training)
        baseline   = np.full(n, capacity_mw)
        blended_fc = 0.70 * prophet_fc + 0.30 * baseline
        method     = "blend_prophet_baseline"

    # ── Capacity baseline ─────────────────────────────────────────
    else:
        blended_fc = np.full(n, capacity_mw, dtype=float)
        fc_lower   = np.maximum(0,           blended_fc - 1.5 * val_rmse)
        fc_upper   = np.minimum(capacity_mw, blended_fc + 1.5 * val_rmse)
        method     = "capacity_baseline"

    # Hard physical bounds
    blended_fc = np.clip(blended_fc, 0, capacity_mw)
    fc_lower   = np.maximum(0,           fc_lower)
    fc_upper   = np.minimum(capacity_mw, fc_upper)

    return pd.DataFrame({
        "plant_id":              plant_id,
        "timestamp":             forecast_timestamps,
        "availability_forecast": np.round(blended_fc, 2),
        "lower_90":              np.round(fc_lower,   2),
        "upper_90":              np.round(fc_upper,   2),
        "method":                method,
    })


# ══════════════════════════════════════════════════════════════════════
# PART 4 — DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════

def prepare_availability_input(merged_df: pd.DataFrame) -> pd.DataFrame:
    needed = ["timestamp", "plant_id", "availability_mw", "capacity_mw"]
    df = merged_df[needed].copy()
    df["timestamp"]       = pd.to_datetime(df["timestamp"])
    df["availability_mw"] = pd.to_numeric(df["availability_mw"], errors="coerce")
    df["capacity_mw"]     = pd.to_numeric(df["capacity_mw"],     errors="coerce")

    parts = []
    for plant_id, plant_df in df.groupby("plant_id"):
        capacity_mw = plant_df["capacity_mw"].median()

        series = (
            plant_df.set_index("timestamp")["availability_mw"]
            .sort_index()
            .reindex(pd.date_range(
                plant_df["timestamp"].min(),
                plant_df["timestamp"].max(), freq="h",
            ))
        )
        series = series.ffill(limit=3)
        series = series.fillna(capacity_mw)
        series = series.clip(lower=0, upper=capacity_mw)

        parts.append(pd.DataFrame({
            "timestamp":       series.index,
            "plant_id":        plant_id,
            "availability_mw": series.values,
            "capacity_mw":     capacity_mw,
        }))

    out = pd.concat(parts, ignore_index=True)
    avg_utilisation = (out["availability_mw"] / out["capacity_mw"]).mean() * 100
    print(f"[Availability prep] {out['plant_id'].nunique()} plants | "
          f"{len(out):,} rows | avg utilisation: {avg_utilisation:.1f}%")
    return out


# ══════════════════════════════════════════════════════════════════════
# PART 5 — FLEET RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_availability_fleet(
    merged_df: pd.DataFrame,
    horizon: int = 72,
    n_jobs: int = -1,
) -> pd.DataFrame:
    import multiprocessing
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    avail_df = prepare_availability_input(merged_df)
    groups   = [(pid, grp.copy()) for pid, grp in avail_df.groupby("plant_id")]
    n        = len(groups)
    resolved = min(multiprocessing.cpu_count() if n_jobs == -1 else n_jobs, n)

    print(f"\n{'═'*55}")
    print(f"  AVAILABILITY FLEET FORECAST")
    print(f"  Plants: {n} | Workers: {resolved} | Horizon: {horizon}h")
    print(f"{'═'*55}")

    results = Parallel(n_jobs=resolved, backend="loky", verbose=5)(
        delayed(_availability_worker)(pid, grp, horizon)
        for pid, grp in groups
    )

    forecasts, failed = [], []
    for r in results:
        if r["status"] == "ok":
            forecasts.append(r["forecast_df"])
            print(f"  ✓ {r['plant_id']} → {r['best_model']}")
        else:
            print(f"  ✗ {r['plant_id']} → {r.get('error', '')[:60]}")
            failed.append(r["plant_id"])

    if not forecasts:
        print("[FATAL] No availability forecasts generated")
        return pd.DataFrame()

    out = pd.concat(forecasts, ignore_index=True)
    out.to_csv(OUT_DIR / "availability_forecasts.csv", index=False)
    print(f"\n  Saved → {OUT_DIR}/availability_forecasts.csv")
    print(f"  Success: {len(forecasts)} | Failed: {len(failed)}")
    return out