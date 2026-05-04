"""
  irradiance_wm2:
    - Strong daily sinusoidal pattern (0 at night, peak at noon)
    - Varies by season (Karnataka summer vs monsoon)
    - Short-term cloud cover adds noise
    → Best models: Prophet (handles zero-at-night cleanly) + physics baseline
    → DO NOT use SARIMA (zero-inflation breaks it)
    → DO NOT use LSTM (overkill for a known physical pattern)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("[WARN] prophet not installed — irradiance will use physics baseline only")

OUT_DIR = Path("data/forecasts")


# ══════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════

def _time_split(series: pd.Series, val_hours: int = 168):
    """Split on time boundary — never shuffle."""
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
    print(f"    {name:20s} → MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.1f}%")
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2)}


# ══════════════════════════════════════════════════════════════════════
# PART 1 — IRRADIANCE FORECAST
# ══════════════════════════════════════════════════════════════════════

"""
IRRADIANCE CHARACTERISTICS:
  - Physical upper bound: ~1200 W/m² (clear sky at noon, Karnataka)
  - Hard lower bound: 0 (night, no negative irradiance)
  - Daily pattern: sin(π × (hour-6)/12) for hours 6–18, else 0
  - Seasonal variation: stronger in summer (Mar–May), weaker in monsoon (Jun–Sep)
  - Cloud cover adds stochastic noise on top of deterministic pattern
  - Zero-inflation: ~50% of hours are exactly 0 (nighttime)

WHAT TO FORECAST:
  We forecast CLEAR-SKY irradiance (the deterministic component).
  Cloud cover modulation is handled separately using weather API.
  Final irradiance = clear_sky_forecast × cloud_correction_factor

  This decomposition is more accurate than forecasting raw irradiance
  because the deterministic part is perfectly known from astronomy.
"""

def _physics_irradiance_forecast(
    timestamps: pd.DatetimeIndex,
    latitude: float = 15.0,
) -> np.ndarray:
    """
    Physics-based clear-sky irradiance estimate.

    Uses simplified sinusoidal model — accurate enough for ML features.
    For production use pvlib for precise solar geometry.

    Formula:
      I_cs = I_0 × max(0, sin(π × (hour - sunrise) / daylight_hours))
      Where I_0 ≈ 950 W/m² (Karnataka clear-sky peak)
            sunrise ≈ 6h, sunset ≈ 18h (varies ±1h seasonally)

    Seasonal adjustment:
      Summer (Mar–May):  × 1.10 (longer days, higher sun angle)
      Monsoon (Jun–Sep): × 0.75 (heavy cloud base reduces clear-sky)
      Winter (Nov–Feb):  × 0.90 (lower sun angle)
    """
    seasonal_factor = {
        1: 0.90, 2: 0.90, 3: 1.10, 4: 1.10, 5: 1.10,
        6: 0.75, 7: 0.75, 8: 0.75, 9: 0.75,
        10: 0.95, 11: 0.90, 12: 0.90,
    }
    irr = []
    for ts in timestamps:
        month  = ts.month
        hour   = ts.hour + ts.minute / 60
        s_fact = seasonal_factor.get(month, 1.0)
        if 6 <= hour <= 18:
            peak = np.sin(np.pi * (hour - 6) / 12)
            val  = max(0, 950 * peak * s_fact)
        else:
            val = 0.0
        irr.append(val)
    return np.array(irr)


def _prophet_irradiance_forecast(
    train: pd.Series,
    horizon: int,
) -> tuple:
    """
    Prophet model for irradiance.

    Key config differences from generation forecast:
    - seasonality_mode='multiplicative': irradiance scales with daylight
    - Strong daily_seasonality with high fourier_order (sharp sunrise/sunset)
    - Non-negative floor enforced (irradiance ≥ 0)
    - logistic growth with cap to respect physical upper bound

    Returns: (model, forecast_df)
    """
    if not HAS_PROPHET:
        raise ImportError("prophet not installed")

    df = train.reset_index()
    df.columns = ["ds", "y"]

    # Enforce non-negative floor for logistic growth
    df["floor"] = 0.0
    df["cap"]   = 1300.0   # physical max irradiance (W/m²)

    model = Prophet(
        growth="logistic",               # respects 0 floor and physical cap
        daily_seasonality=False,         # we'll add a custom one below
        weekly_seasonality=False,        # irradiance has no weekly pattern
        yearly_seasonality=True,         # seasonal cloud/sun angle variation
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.001,   # irradiance trend is very stable
        interval_width=0.90,
    )

    # Custom high-resolution daily seasonality
    # fourier_order=12 gives sharp transitions at sunrise/sunset
    model.add_seasonality(
        name="daily_solar",
        period=1,
        fourier_order=12,
        mode="multiplicative",
    )

    model.fit(df)

    future = model.make_future_dataframe(periods=horizon, freq="h")
    future["floor"] = 0.0
    future["cap"]   = 1300.0

    forecast = model.predict(future)
    # Hard clip to physical bounds
    forecast["yhat"] = forecast["yhat"].clip(lower=0, upper=1300)
    return model, forecast.tail(horizon).reset_index(drop=True)


def forecast_irradiance_one_plant(
    plant_id: str,
    series: pd.Series,                  # hourly irradiance_wm2 with DatetimeIndex
    plant_meta: dict,                   # {latitude, longitude, region}
    horizon: int = 72,
    val_hours: int = 168,
) -> dict:
    """
    Forecast irradiance_wm2 for one plant for the next `horizon` hours.

    Strategy:
      1. Physics baseline  — deterministic, always available
      2. Prophet           — learns cloud cover patterns from history
      3. Pick winner on MAE on validation set
      4. Blend: winner × 0.7 + physics × 0.3 (physics anchors the forecast)

    Why blend instead of pure winner:
      Prophet can drift on cloudy periods.
      Physics baseline is always correct on the time-of-day shape.
      Blending gets the shape right AND the magnitude right.

    Returns: dict with forecast_df (timestamp, irradiance_forecast,
             lower_90, upper_90, method)
    """
    print(f"\n  [{plant_id}] Irradiance forecast ({len(series):,} obs)...")

    series = series.clip(lower=0)       # physical constraint
    lat    = plant_meta.get("latitude", 15.0)
    train, val = _time_split(series, val_hours)

    results = {}

    # ── 1. Physics baseline ───────────────────────────────────────────
    physics_val = _physics_irradiance_forecast(val.index, latitude=lat)
    results["physics"] = _evaluate(val.values, physics_val, "physics baseline")

    # ── 2. Prophet ────────────────────────────────────────────────────
    prophet_val_fc = None
    if HAS_PROPHET:
        try:
            _, prophet_val_df = _prophet_irradiance_forecast(train, horizon=val_hours)
            prophet_val_fc    = prophet_val_df["yhat"].values
            results["prophet"] = _evaluate(val.values, prophet_val_fc, "prophet")
        except Exception as e:
            print(f"    Prophet failed: {e}")

    # ── Pick winner ───────────────────────────────────────────────────
    best = min(results, key=lambda m: results[m]["MAE"])
    print(f"    Winner: {best} (MAE={results[best]['MAE']:.3f})")

    # ── Forecast on full series ───────────────────────────────────────
    forecast_timestamps = pd.date_range(
        start=series.index[-1] + pd.Timedelta(hours=1),
        periods=horizon, freq="h",
    )

    # Physics forecast — always compute (used for blend)
    physics_fc = _physics_irradiance_forecast(forecast_timestamps, latitude=lat)

    if best == "prophet" and HAS_PROPHET:
        _, fc_df_full = _prophet_irradiance_forecast(series, horizon=horizon)
        prophet_fc    = fc_df_full["yhat"].values
        fc_lower      = fc_df_full["yhat_lower"].clip(lower=0).values
        fc_upper      = fc_df_full["yhat_upper"].values
        # Blend: Prophet for magnitude, physics for shape
        blended_fc    = 0.70 * prophet_fc + 0.30 * physics_fc
    else:
        blended_fc = physics_fc
        rmse       = results["physics"]["RMSE"]
        fc_lower   = np.maximum(0, blended_fc - 1.5 * rmse)
        fc_upper   = blended_fc + 1.5 * rmse

    # Hard physical constraints
    blended_fc = np.clip(blended_fc, 0, 1300)
    fc_lower   = np.maximum(0, fc_lower)
    fc_upper   = np.minimum(1300, fc_upper)

    # Zero out night hours — physics must win here
    for i, ts in enumerate(forecast_timestamps):
        if ts.hour < 6 or ts.hour > 18:
            blended_fc[i] = 0.0
            fc_lower[i]   = 0.0
            fc_upper[i]   = 0.0

    forecast_df = pd.DataFrame({
        "plant_id":            plant_id,
        "timestamp":           forecast_timestamps,
        "irradiance_forecast": np.round(blended_fc, 2),
        "lower_90":            np.round(fc_lower,   2),
        "upper_90":            np.round(fc_upper,   2),
        "method":              f"blend_{best}_physics",
    })

    print(f"    Daytime peak: {blended_fc.max():.1f} W/m²  "
          f"| Night hours zeroed: {(blended_fc == 0).sum()}")

    return {
        "plant_id":    plant_id,
        "scores":      results,
        "best_model":  best,
        "forecast_df": forecast_df,
    }


# ══════════════════════════════════════════════════════════════════════
# PART 3 — DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════

def prepare_irradiance_input(
    merged_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare irradiance series for forecasting.

    Input : merged_df with timestamp, plant_id, irradiance_wm2
    Output: clean df with timestamp, plant_id, irradiance (renamed)
            — physically bounded, gaps filled, sorted
    """
    df = merged_df[["timestamp", "plant_id", "irradiance_wm2"]].copy()
    df["timestamp"]     = pd.to_datetime(df["timestamp"])
    df["irradiance_wm2"] = pd.to_numeric(df["irradiance_wm2"], errors="coerce")

    df = df.rename(columns={"irradiance_wm2": "irradiance"})

    parts = []
    for plant_id, plant_df in df.groupby("plant_id"):
        series = (
            plant_df.set_index("timestamp")["irradiance"]
            .sort_index()
            .reindex(pd.date_range(
                plant_df["timestamp"].min(),
                plant_df["timestamp"].max(), freq="h"
            ))
        )
        # Night hours should be 0, not NaN — distinguish truly missing from night
        night_mask = (series.index.hour < 6) | (series.index.hour > 18)
        series[night_mask] = series[night_mask].fillna(0)
        # Day gaps: interpolate up to 3 hours
        series = series.interpolate(method="linear", limit=3).fillna(0)
        series = series.clip(lower=0, upper=1300)

        parts.append(pd.DataFrame({
            "timestamp": series.index,
            "plant_id":  plant_id,
            "irradiance": series.values,
        }))

    out = pd.concat(parts, ignore_index=True)
    print(f"[Irradiance prep] {out['plant_id'].nunique()} plants | "
          f"{len(out):,} rows | "
          f"daytime mean: {out[out['irradiance']>0]['irradiance'].mean():.1f} W/m²")
    return out


# ══════════════════════════════════════════════════════════════════════
# PART 4 — FLEET RUNNERS (parallelised)
# ══════════════════════════════════════════════════════════════════════
import joblib
from pathlib import Path
from datetime import datetime

MODEL_DIR = Path("models/irradiance")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def _irradiance_worker(plant_id, plant_df, plant_meta, horizon):
    warnings.filterwarnings("ignore")
    try:
        series = (
            plant_df.sort_values("timestamp")
            .set_index("timestamp")["irradiance"]
            .asfreq("h").fillna(0)
        )
        if len(series) < 500:
            return {"status": "skip", "plant_id": plant_id,
                    "error": f"only {len(series)} obs"}
        result = forecast_irradiance_one_plant(
            plant_id, series, plant_meta, horizon
        )

        # ── Save model ───────────────────────────────
        model = result.get("best_model")
        if model is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            model_path = MODEL_DIR / f"{plant_id}_irradiance_model_{ts}.joblib"

            joblib.dump(
                {
                    "model": model,
                    "plant_id": plant_id,
                    "plant_meta": plant_meta,
                    "horizon": horizon,
                    "model_name": result.get("best_model"),
                },
                model_path,
                compress=3,
            )
        return {"status": "ok", "plant_id": plant_id,
                "forecast_df": result["forecast_df"],
                "best_model": result["best_model"]}
    except Exception as e:
        import traceback
        return {"status": "error", "plant_id": plant_id,
                "forecast_df": None,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}



def run_irradiance_fleet(
    merged_df: pd.DataFrame,
    plant_master_df: pd.DataFrame,
    horizon: int = 72,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Forecast irradiance_wm2 for all plants in parallel.

    Parameters
    ----------
    merged_df       : has timestamp, plant_id, irradiance_wm2
    plant_master_df : has plant_id, latitude, longitude, region
    horizon         : forecast hours (default 72)
    n_jobs          : parallel workers

    Returns pd.DataFrame with irradiance forecasts for all plants
    """
    import multiprocessing
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    irr_df   = prepare_irradiance_input(merged_df)
    meta_map = plant_master_df.set_index("plant_id").to_dict("index")

    groups = [
        (pid, grp.copy(), meta_map.get(pid, {"latitude": 15.0}))
        for pid, grp in irr_df.groupby("plant_id")
    ]
    n        = len(groups)
    resolved = min(multiprocessing.cpu_count() if n_jobs == -1 else n_jobs, n)

    print(f"\n{'═'*55}")
    print(f"  IRRADIANCE FLEET FORECAST")
    print(f"  Plants: {n} | Workers: {resolved} | Horizon: {horizon}h")
    print(f"{'═'*55}")

    results = Parallel(n_jobs=resolved, backend="loky", verbose=5)(
        delayed(_irradiance_worker)(pid, grp, meta, horizon)
        for pid, grp, meta in groups
    )

    forecasts, failed = [], []
    for r in results:
        if r["status"] == "ok":
            forecasts.append(r["forecast_df"])
            print(f"  ✓ {r['plant_id']} → {r['best_model']}")
        else:
            print(f"  ✗ {r['plant_id']} → {r.get('error','')[:60]}")
            failed.append(r["plant_id"])

    if not forecasts:
        print("[FATAL] No irradiance forecasts generated")
        return pd.DataFrame()

    out = pd.concat(forecasts, ignore_index=True)
    out.to_csv(OUT_DIR / "irradiance_forecasts.csv", index=False)
    print(f"\n  Saved → {OUT_DIR}/irradiance_forecasts.csv")
    print(f"  Success: {len(forecasts)} | Failed: {len(failed)}")
    return out
