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
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("[WARN] prophet not installed — availability will use capacity baseline only")

OUT_DIR = Path("data/forecasts")


# ══════════════════════════════════════════════════════════════════════
# SHARED UTILITIES  (mirrors irradiance.py exactly)
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
    print(f"    {name:25s} → MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.1f}%")
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2)}


# ══════════════════════════════════════════════════════════════════════
# PART 1 — AVAILABILITY FORECAST
# ══════════════════════════════════════════════════════════════════════

"""
AVAILABILITY CHARACTERISTICS:
  - Hard upper bound: capacity_mw (never exceeds installed capacity)
  - Hard lower bound: 0 (full outage)
  - Typical pattern: stays near capacity_mw, drops sharply during outages,
    recovers immediately after maintenance completes
  - Weekly cycle: planned maintenance often on weekends / fixed days
  - Yearly cycle: monsoon season (Jun–Sep) sees more maintenance windows
                  in Karnataka (lower generation → good time to maintain)
  - Zero-inflation: forced outages cause sudden zeros (not predictable,
    but rolling history captures plant-specific reliability patterns)

WHAT TO FORECAST:
  We forecast availability_mw directly (not as a fraction).
  Capacity baseline: assume full availability = capacity_mw.
  Prophet learns departures from full availability (planned outages,
  maintenance windows) from historical patterns.

  Final forecast is clipped to [0, capacity_mw] per plant.
"""


def _capacity_baseline_forecast(
    capacity_mw: float,
    n_periods: int,
) -> np.ndarray:
    """
    Simplest possible baseline: assume full availability at all times.

    This is surprisingly strong for plants with high reliability (>95% uptime).
    Acts as the upper-bound anchor in the blend, equivalent to how
    physics_irradiance anchors the irradiance forecast.

    Parameters
    ----------
    capacity_mw : float   installed capacity of the plant
    n_periods   : int     number of hourly timesteps to fill
    """
    return np.full(n_periods, capacity_mw, dtype=float)


def _prophet_availability_forecast(
    train: pd.Series,
    capacity_mw: float,
    horizon: int,
) -> tuple:
    """
    Prophet model for availability_mw.

    Key config differences from irradiance:
    - seasonality_mode='additive': outage drops are absolute MW, not scaled
    - weekly_seasonality=True: maintenance has strong day-of-week pattern
    - yearly_seasonality=True: monsoon maintenance windows
    - No daily_seasonality: availability doesn't follow a daily solar arc
    - logistic growth with cap=capacity_mw to respect physical upper bound
    - changepoint_prior_scale higher than irradiance (outages are abrupt)

    Returns: (model, forecast_df)
    """
    if not HAS_PROPHET:
        raise ImportError("prophet not installed")

    df = train.reset_index()
    df.columns = ["ds", "y"]

    # Logistic growth keeps forecast ≤ capacity_mw
    df["floor"] = 0.0
    df["cap"]   = float(capacity_mw)

    model = Prophet(
        growth="logistic",
        daily_seasonality=False,          # no daily arc unlike irradiance
        weekly_seasonality=True,          # maintenance has weekly pattern
        yearly_seasonality=True,          # monsoon maintenance windows
        seasonality_mode="additive",      # outage drops are in MW not fractions
        changepoint_prior_scale=0.05,     # more flexible than irradiance (abrupt outages)
        seasonality_prior_scale=10.0,     # allow stronger weekly/yearly swings
        interval_width=0.90,
    )

    # Add a custom monthly seasonality to capture maintenance planning cycles
    model.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=5,
        mode="additive",
    )

    model.fit(df)

    future = model.make_future_dataframe(periods=horizon, freq="h")
    future["floor"] = 0.0
    future["cap"]   = float(capacity_mw)

    forecast = model.predict(future)
    # Hard clip to physical bounds
    forecast["yhat"] = forecast["yhat"].clip(lower=0, upper=capacity_mw)
    return model, forecast.tail(horizon).reset_index(drop=True)


def forecast_availability_one_plant(
    plant_id: str,
    series: pd.Series,           # hourly availability_mw with DatetimeIndex
    capacity_mw: float,          # installed capacity — hard upper bound
    horizon: int = 72,
    val_hours: int = 168,
) -> dict:
    """
    Forecast availability_mw for one plant for the next `horizon` hours.

    Strategy (mirrors irradiance.py):
      1. Capacity baseline — assume full availability (always available)
      2. Prophet           — learns planned outage / maintenance patterns
      3. Pick winner on MAE on validation set
      4. Blend: winner × 0.7 + capacity_baseline × 0.3
         (baseline anchors forecast to capacity, Prophet adjusts for outages)

    Why blend:
      Prophet can over-predict outage depth on unseen periods.
      Capacity baseline is always a reasonable upper anchor.
      Blending gives conservative, stable forecasts for grid planning.

    Physical constraints enforced:
      - forecast clipped to [0, capacity_mw]
      - no negative availability

    Returns
    -------
    dict with keys: plant_id, scores, best_model, forecast_df
    forecast_df columns: plant_id, timestamp, availability_forecast,
                         lower_90, upper_90, method
    """
    print(f"\n  [{plant_id}] Availability forecast ({len(series):,} obs, "
          f"capacity={capacity_mw:.1f} MW)...")

    # Physical constraint — availability can't exceed capacity
    series = series.clip(lower=0, upper=capacity_mw)
    train, val = _time_split(series, val_hours)

    results = {}

    # ── 1. Capacity baseline ──────────────────────────────────────────
    baseline_val = _capacity_baseline_forecast(capacity_mw, len(val))
    results["capacity_baseline"] = _evaluate(val.values, baseline_val, "capacity baseline")

    # ── 2. Prophet ────────────────────────────────────────────────────
    prophet_val_fc = None
    if HAS_PROPHET:
        try:
            _, prophet_val_df = _prophet_availability_forecast(
                train, capacity_mw, horizon=val_hours
            )
            prophet_val_fc     = prophet_val_df["yhat"].values
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

    # Capacity baseline — always compute (used for blend)
    baseline_fc = _capacity_baseline_forecast(capacity_mw, horizon)

    if best == "prophet" and HAS_PROPHET:
        _, fc_df_full = _prophet_availability_forecast(series, capacity_mw, horizon)
        prophet_fc    = fc_df_full["yhat"].values
        fc_lower      = fc_df_full["yhat_lower"].clip(lower=0).values
        fc_upper      = fc_df_full["yhat_upper"].clip(upper=capacity_mw).values
        # Blend: Prophet for outage patterns, baseline for capacity anchor
        blended_fc    = 0.70 * prophet_fc + 0.30 * baseline_fc
    else:
        blended_fc = baseline_fc
        rmse       = results["capacity_baseline"]["RMSE"]
        fc_lower   = np.maximum(0, blended_fc - 1.5 * rmse)
        fc_upper   = np.minimum(capacity_mw, blended_fc + 1.5 * rmse)

    # Hard physical constraints
    blended_fc = np.clip(blended_fc, 0, capacity_mw)
    fc_lower   = np.maximum(0, fc_lower)
    fc_upper   = np.minimum(capacity_mw, fc_upper)

    forecast_df = pd.DataFrame({
        "plant_id":               plant_id,
        "timestamp":              forecast_timestamps,
        "availability_forecast":  np.round(blended_fc, 2),
        "lower_90":               np.round(fc_lower,   2),
        "upper_90":               np.round(fc_upper,   2),
        "method":                 f"blend_{best}_baseline",
    })

    uptime_pct = (blended_fc > 0).mean() * 100
    avg_avail  = blended_fc.mean()
    print(f"    Avg forecast: {avg_avail:.1f} MW  "
          f"| Uptime: {uptime_pct:.1f}%  "
          f"| Capacity: {capacity_mw:.1f} MW")

    return {
        "plant_id":    plant_id,
        "scores":      results,
        "best_model":  best,
        "forecast_df": forecast_df,
    }


# ══════════════════════════════════════════════════════════════════════
# PART 3 — DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════

def prepare_availability_input(
    merged_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare availability_mw series for forecasting.

    Input : merged_df with timestamp, plant_id, availability_mw, capacity_mw
    Output: clean df with timestamp, plant_id, availability_mw, capacity_mw
            — physically bounded [0, capacity_mw], gaps forward-filled,
              sorted per plant

    Gap-filling strategy:
      - Short gaps (≤ 3h): forward-fill (availability held constant
        until next reading — matches how SCADA systems report it)
      - Longer gaps: fill with capacity_mw (assume available unless told otherwise)
      This is conservative — better to over-estimate availability
      than to inject phantom outages into training data.
    """
    needed = ["timestamp", "plant_id", "availability_mw", "capacity_mw"]
    df = merged_df[needed].copy()
    df["timestamp"]       = pd.to_datetime(df["timestamp"])
    df["availability_mw"] = pd.to_numeric(df["availability_mw"], errors="coerce")
    df["capacity_mw"]     = pd.to_numeric(df["capacity_mw"],     errors="coerce")

    parts = []
    for plant_id, plant_df in df.groupby("plant_id"):
        capacity_mw = plant_df["capacity_mw"].median()   # stable plant property

        series = (
            plant_df.set_index("timestamp")["availability_mw"]
            .sort_index()
            .reindex(pd.date_range(
                plant_df["timestamp"].min(),
                plant_df["timestamp"].max(), freq="h",
            ))
        )

        # Forward-fill short gaps (SCADA holds last value)
        series = series.ffill(limit=3)
        # Remaining NaNs → assume full availability
        series = series.fillna(capacity_mw)
        # Physical bounds
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
          f"{len(out):,} rows | "
          f"avg utilisation: {avg_utilisation:.1f}%")
    return out


# ══════════════════════════════════════════════════════════════════════
# PART 4 — FLEET RUNNERS (parallelised)
# ══════════════════════════════════════════════════════════════════════

def _availability_worker(plant_id, plant_df, horizon):
    warnings.filterwarnings("ignore")
    try:
        capacity_mw = plant_df["capacity_mw"].median()
        series = (
            plant_df.sort_values("timestamp")
            .set_index("timestamp")["availability_mw"]
            .asfreq("h")
            .fillna(capacity_mw)      # assume available when reading missing
            .clip(lower=0, upper=capacity_mw)
        )
        if len(series) < 500:
            return {"status": "skip", "plant_id": plant_id,
                    "error": f"only {len(series)} obs"}
        result = forecast_availability_one_plant(
            plant_id, series, capacity_mw, horizon
        )
        return {"status": "ok", "plant_id": plant_id,
                "forecast_df": result["forecast_df"],
                "best_model":  result["best_model"]}
    except Exception as e:
        import traceback
        return {"status": "error", "plant_id": plant_id,
                "forecast_df": None,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


def run_availability_fleet(
    merged_df: pd.DataFrame,
    horizon: int = 72,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Forecast availability_mw for all plants in parallel.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Must contain: timestamp, plant_id, availability_mw, capacity_mw
    horizon   : int
        Forecast hours (default 72 — matches irradiance fleet default)
    n_jobs    : int
        Parallel workers (-1 = all cores)

    Returns
    -------
    pd.DataFrame with columns:
        plant_id, timestamp, availability_forecast, lower_90, upper_90, method
    Also saves → data/forecasts/availability_forecasts.csv
    """
    import multiprocessing
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    avail_df = prepare_availability_input(merged_df)

    groups = [
        (pid, grp.copy())
        for pid, grp in avail_df.groupby("plant_id")
    ]
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