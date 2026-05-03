"""
auxiliary_forecasts.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Univariate forecasts for irradiance_wm2 and health_factor.

WHY SEPARATE FROM GENERATION FORECAST:
  At t+1..t+72, three things are unknown:
    1. generation     → already handled in univariate.py
    2. irradiance_wm2 → handled HERE (physics-driven, daily cycle)
    3. health_factor  → handled HERE (slow degradation, step changes)

  Both are used as features in the multivariate model.
  If you carry-forward their last known value instead of forecasting:
    - irradiance: every forecast day looks like yesterday's cloud cover
    - health_factor: you miss scheduled repairs in the forecast window

DIFFERENT MODELS PER VARIABLE:
  irradiance_wm2:
    - Strong daily sinusoidal pattern (0 at night, peak at noon)
    - Varies by season (Karnataka summer vs monsoon)
    - Short-term cloud cover adds noise
    → Best models: Prophet (handles zero-at-night cleanly) + physics baseline
    → DO NOT use SARIMA (zero-inflation breaks it)
    → DO NOT use LSTM (overkill for a known physical pattern)

  health_factor:
    - Slow monotone degradation (−0.3% to −0.5% per month)
    - Occasional step INCREASES (repair/maintenance events)
    - Series is nearly constant over 72 hours
    → Best models: Linear trend extrapolation + change point detection
    → DO NOT use Prophet (interprets step changes as anomalies)
    → DO NOT use SARIMA (no seasonality to model)
    → Simple is better here — linear regression beats complex models
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
# PART 2 — HEALTH FACTOR FORECAST
# ══════════════════════════════════════════════════════════════════════

"""
HEALTH FACTOR CHARACTERISTICS:
  - Range: 0.35 (severely degraded) to 0.99 (post-repair)
  - Slowly decreasing over time: −0.003 to −0.006 per month
  - Occasional step INCREASES: repair events restore +0.15 to +0.35
  - Over 72 hours: changes are tiny (max ~0.014 degradation in 3 days)
  - No daily/weekly seasonality — purely time-driven process

  Key insight: over 72 hours, health_factor barely changes UNLESS
  a scheduled repair happens in the forecast window.
  → Forecasting degradation is trivial (linear extrapolation)
  → Forecasting REPAIR EVENTS is the hard part

WHAT TO FORECAST:
  1. Degradation trend: linear regression on recent history
  2. Repair probability: logistic regression on time-since-last-repair,
     current health level, and maintenance schedule patterns
  3. Final forecast = degradation_trend + repair_adjustment
"""

def _detect_repair_events(series: pd.Series) -> pd.DataFrame:
    """
    Detect historical repair events from health_factor series.

    A repair event is identified as any hour where health_factor
    increases by more than 0.05 from the previous hour.
    (Normal degradation is −0.0001 per hour, so +0.05 is a clear signal)

    Returns DataFrame of repair events with columns:
        timestamp, health_before, health_after, magnitude
    """
    diff   = series.diff()
    repair = diff[diff > 0.05]   # threshold: >5% jump = repair event

    events = []
    for ts, magnitude in repair.items():
        events.append({
            "timestamp":     ts,
            "health_before": float(series[ts] - magnitude),
            "health_after":  float(series[ts]),
            "magnitude":     round(float(magnitude), 4),
        })

    return pd.DataFrame(events) if events else pd.DataFrame(
        columns=["timestamp", "health_before", "health_after", "magnitude"]
    )


def _estimate_repair_probability(
    series: pd.Series,
    repair_events: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    """
    Estimate probability of a repair event in the next `horizon` hours.

    Logic:
      - If health is below 0.60 and more than 30 days since last repair
        → probability of repair increases linearly with time
      - Historical repair frequency gives the base rate
      - Returns an array of per-hour repair probabilities

    This is intentionally simple — repair scheduling is a business process
    not fully captured in the data. The probability adds uncertainty to CI.
    """
    current_health = float(series.iloc[-1])
    n_hours        = len(series)

    # Base repair rate from history (repairs per hour)
    n_repairs  = len(repair_events)
    base_rate  = n_repairs / n_hours if n_hours > 0 else 0.001

    # Health urgency factor: lower health → higher repair probability
    if current_health < 0.55:
        urgency = 3.0    # critical — repair very likely
    elif current_health < 0.65:
        urgency = 2.0    # degraded — repair likely soon
    elif current_health < 0.75:
        urgency = 1.0    # moderate — normal maintenance cycle
    else:
        urgency = 0.3    # healthy — unlikely to repair

    # Time since last repair (in hours)
    if len(repair_events) > 0:
        last_repair = pd.Timestamp(repair_events["timestamp"].max())
        hours_since = (series.index[-1] - last_repair).total_seconds() / 3600
    else:
        hours_since = n_hours   # assume no repair ever done

    # Repair probability increases if long time since last repair
    time_factor = min(2.0, hours_since / (30 * 24))   # caps at 2× after 30 days

    hourly_prob = min(0.05, base_rate * urgency * time_factor)

    # Returns same probability each hour (simplified model)
    return np.full(horizon, hourly_prob)


def forecast_health_factor_one_plant(
    plant_id: str,
    series: pd.Series,              # hourly health_factor with DatetimeIndex
    lifecycle_df: pd.DataFrame,     # lifecycle_events.csv filtered to this plant
    horizon: int = 72,
    val_hours: int = 168,
) -> dict:
    """
    Forecast health_factor for one plant for the next `horizon` hours.

    Strategy:
      Stage 1 — Degradation trend:
        Linear regression on last 30 days of health_factor.
        Extrapolate slope forward for 72 hours.
        This captures the current degradation rate.

      Stage 2 — Repair adjustment:
        Detect historical repair events.
        Estimate probability of repair in forecast window.
        Widen confidence interval to reflect repair uncertainty.

      Stage 3 — Boundary constraints:
        Health cannot exceed 0.99 (even after repair).
        Health cannot go below 0.35 (plant would be offline).
        Repair events are modelled as discrete upward steps.

    Returns: dict with forecast_df (timestamp, health_forecast,
             lower_90, upper_90, repair_probability)
    """
    print(f"\n  [{plant_id}] Health factor forecast ({len(series):,} obs)...")

    series = series.clip(0.35, 0.99)   # physical bounds
    train, val = _time_split(series, val_hours)

    # ── Stage 1: Linear degradation trend ────────────────────────────
    # Use last 30 days for trend estimation (not entire history —
    # earlier periods may have different degradation rates)
    window = min(len(train), 30 * 24)
    recent = train.iloc[-window:]

    t  = np.arange(len(recent)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(t, recent.values)

    slope         = float(lr.coef_[0])         # degradation per hour
    current_level = float(series.iloc[-1])

    print(f"    Current health : {current_level:.4f}")
    print(f"    Hourly slope   : {slope:.6f}  "
          f"({slope * 24 * 30:.4f} per month)")

    # ── Stage 1 validation ────────────────────────────────────────────
    t_val     = np.arange(len(recent), len(recent) + val_hours).reshape(-1, 1)
    val_pred  = lr.predict(t_val).flatten()
    val_pred  = np.clip(val_pred, 0.35, 0.99)
    val_score = _evaluate(val.values, val_pred, "linear degradation")

    # ── Stage 2: Repair event detection ──────────────────────────────
    repair_events = _detect_repair_events(series)
    print(f"    Repair events detected: {len(repair_events)}")
    if len(repair_events) > 0:
        avg_magnitude = repair_events["magnitude"].mean()
        print(f"    Avg repair magnitude: +{avg_magnitude:.4f}")

    repair_probs = _estimate_repair_probability(series, repair_events, horizon)
    max_repair_prob = repair_probs.max()
    print(f"    Repair probability (per hour): {max_repair_prob:.4f}")

    # ── Stage 3: Forecast ─────────────────────────────────────────────
    forecast_timestamps = pd.date_range(
        start=series.index[-1] + pd.Timedelta(hours=1),
        periods=horizon, freq="h",
    )

    t_future = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
    fc_trend = lr.predict(t_future).flatten()

    # Expected repair boost (probability × avg magnitude)
    if len(repair_events) > 0:
        avg_magnitude = float(repair_events["magnitude"].mean())
    else:
        avg_magnitude = 0.20    # default repair magnitude if no history

    fc_health = fc_trend.copy()
    for i in range(horizon):
        # Expected value adjustment for repair probability
        fc_health[i] += repair_probs[i] * avg_magnitude

    fc_health = np.clip(fc_health, 0.35, 0.99)

    # Confidence interval
    # — Narrow for degradation (very predictable)
    # — Wider where repair probability is high (uncertain step)
    base_sigma  = val_score["RMSE"]
    repair_sigma = repair_probs * avg_magnitude  # uncertainty from repair
    total_sigma  = np.sqrt(base_sigma ** 2 + repair_sigma ** 2)

    fc_lower = np.maximum(0.35, fc_health - 1.645 * total_sigma)
    fc_upper = np.minimum(0.99, fc_health + 1.645 * total_sigma)

    forecast_df = pd.DataFrame({
        "plant_id":           plant_id,
        "timestamp":          forecast_timestamps,
        "health_forecast":    np.round(fc_health,      4),
        "lower_90":           np.round(fc_lower,       4),
        "upper_90":           np.round(fc_upper,       4),
        "repair_probability": np.round(repair_probs,   4),
        "degradation_slope":  round(slope,             6),
    })

    print(f"    Forecast range: [{fc_health.min():.4f}, {fc_health.max():.4f}]  "
          f"| Max repair risk: {repair_probs.max():.4f}")

    return {
        "plant_id":     plant_id,
        "scores":       {"linear": val_score},
        "repair_events": repair_events,
        "forecast_df":  forecast_df,
    }


# ══════════════════════════════════════════════════════════════════════
# PART 3 — DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════

def prepare_health_input(
    merged_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare health_factor series for forecasting.

    Input : merged_df with timestamp, plant_id, health_factor
    Output: clean df with timestamp, plant_id, health_factor
            — bounded to [0.35, 0.99], monotone-aware gap fill
    """
    df = merged_df[["timestamp", "plant_id", "health_factor"]].copy()
    df["timestamp"]     = pd.to_datetime(df["timestamp"])
    df["health_factor"] = pd.to_numeric(df["health_factor"], errors="coerce")

    parts = []
    for plant_id, plant_df in df.groupby("plant_id"):
        series = (
            plant_df.set_index("timestamp")["health_factor"]
            .sort_index()
            .reindex(pd.date_range(
                plant_df["timestamp"].min(),
                plant_df["timestamp"].max(), freq="h"
            ))
        )
        # Health changes slowly — forward fill up to 24h is safe
        series = series.ffill(limit=24).bfill(limit=6).fillna(0.85)
        series = series.clip(0.35, 0.99)

        parts.append(pd.DataFrame({
            "timestamp":    series.index,
            "plant_id":     plant_id,
            "health_factor": series.values,
        }))

    out = pd.concat(parts, ignore_index=True)
    print(f"[Health prep] {out['plant_id'].nunique()} plants | "
          f"{len(out):,} rows | "
          f"fleet mean health: {out['health_factor'].mean():.4f}")
    return out


# ══════════════════════════════════════════════════════════════════════
# PART 4 — FLEET RUNNERS (parallelised)
# ══════════════════════════════════════════════════════════════════════
import joblib
from pathlib import Path
from datetime import datetime

MODEL_DIR = Path("models/health factor")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def _health_worker(plant_id, plant_df, lifecycle_df, horizon):
    warnings.filterwarnings("ignore")
    try:
        series = (
            plant_df.sort_values("timestamp")
            .set_index("timestamp")["health_factor"]
            .asfreq("h").ffill(limit=24).fillna(0.85)
        )
        if len(series) < 500:
            return {"status": "skip", "plant_id": plant_id,
                    "error": f"only {len(series)} obs"}
        plant_lc = lifecycle_df[lifecycle_df["plant_id"] == plant_id]
        result   = forecast_health_factor_one_plant(
            plant_id, series, plant_lc, horizon
        )

        # ── Save model ───────────────────────────────
        model = result.get("model", None)
        if model is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            model_path = MODEL_DIR / f"{plant_id}_health_model_{ts}.joblib"

            joblib.dump(
                {
                    "model": model,
                    "plant_id": plant_id,
                    "horizon": horizon,
                },
                model_path,
                compress=3,
            )
        return {"status": "ok", "plant_id": plant_id,
                "forecast_df": result["forecast_df"],
                "repair_events": result["repair_events"]}
    except Exception as e:
        import traceback
        return {"status": "error", "plant_id": plant_id,
                "forecast_df": None,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


def run_health_fleet(
    merged_df: pd.DataFrame,
    lifecycle_df: pd.DataFrame,
    horizon: int = 72,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Forecast health_factor for all plants in parallel.

    Parameters
    ----------
    merged_df    : has timestamp, plant_id, health_factor
    lifecycle_df : lifecycle_events.csv — used for repair event history
    horizon      : forecast hours (default 72)
    n_jobs       : parallel workers

    Returns pd.DataFrame with health forecasts for all plants
    """
    import multiprocessing
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    health_df = prepare_health_input(merged_df)

    groups = [
        (pid, grp.copy())
        for pid, grp in health_df.groupby("plant_id")
    ]
    n        = len(groups)
    resolved = min(multiprocessing.cpu_count() if n_jobs == -1 else n_jobs, n)

    print(f"\n{'═'*55}")
    print(f"  HEALTH FACTOR FLEET FORECAST")
    print(f"  Plants: {n} | Workers: {resolved} | Horizon: {horizon}h")
    print(f"{'═'*55}")

    results = Parallel(n_jobs=resolved, backend="loky", verbose=5)(
        delayed(_health_worker)(pid, grp, lifecycle_df, horizon)
        for pid, grp in groups
    )

    forecasts, failed = [], []
    for r in results:
        if r["status"] == "ok":
            forecasts.append(r["forecast_df"])
            n_repair = len(r.get("repair_events", []))
            print(f"  ✓ {r['plant_id']} → {n_repair} historical repair events")
        else:
            print(f"  ✗ {r['plant_id']} → {r.get('error','')[:60]}")
            failed.append(r["plant_id"])

    if not forecasts:
        print("[FATAL] No health forecasts generated")
        return pd.DataFrame()

    out = pd.concat(forecasts, ignore_index=True)
    out.to_csv(OUT_DIR / "health_forecasts.csv", index=False)
    print(f"\n  Saved → {OUT_DIR}/health_forecasts.csv")
    print(f"  Success: {len(forecasts)} | Failed: {len(failed)}")
    return out