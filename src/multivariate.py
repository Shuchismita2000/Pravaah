"""
multivariate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Karnataka Renewable Energy Grid — Multivariate Forecasting Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COLUMN SCHEMA (from historical_df_solar / forecast_df_solar)
─────────────────────────────────────────────────────────────────────
TARGET:   generation (MW)

AVAILABLE IN BOTH HISTORICAL AND FORECAST DFs:
  Identity    : plant_id, capacity_mw, timestamp
  Operational : actual_generation_mw, availability_mw, curtailment_mw
                net_availability_mw, health_factor, status
  Weather     : temperature, cloud_cover, wind_speed, wind_direction
                irradiance, clear_sky_irradiance, irradiance_adjusted
                irradiance_ratio, temp_effect
  Plant type  : plant_type_code, plant_type_Hybrid, plant_type_Solar
                plant_type_Wind
  Derived gen : generation, capacity_factor, generation_shortfall_mw
                generation_norm, health_adjusted_capacity_mw
                adjusted_generation_signal, expected_generation
                performance_ratio
  Flags       : is_degraded, is_offline, is_daylight, is_peak
                is_weekend, is_zero_gen, zero_streak
  Calendar    : hour, day_of_year, month, day_of_week
                hour_sin, hour_cos, doy_sin, doy_cos
                month_sin, month_cos
  Maintenance : days_since_cleaning, soiling_loss
  Lags        : gen_lag_1, gen_lag_24, gen_lag_168
  Rolling     : gen_rolling_mean_6, gen_rolling_std_6
                gen_rolling_mean_24, gen_rolling_mean_168
                pr_rolling_7, pr_rolling_30
  Momentum    : ramp_rate, ramp_abs, gen_momentum_3, gen_momentum_6
  Load/CUF    : cuf, load_factor, daily_generation
  Variability : gen_variability_24, gen_variability_168
  Residual    : gen_residual_24, gen_normalized

DATA FLOW
─────────────────────────────────────────────────────────────────────
HISTORICAL (known):
  historical_df   → all columns above (actual history, is_forecast=False)
  forecast_df     → all columns above for t+1..t+72  (is_forecast=True)
                    generation = univariate forecast_mw proxy
  weather_df      → optional richer weather for t+1..t+72
                    (from OpenWeatherMap; otherwise carry-forward)

WHAT THIS FILE DOES:
  1. Merge         — align historical + forecast dfs into one extended df
  2. Decompose     — STL: trend + seasonal + residual per plant
  3. Feature build — exogenous + endogenous feature matrices
  4. Model         — LightGBM, XGBoost, Ridge, SVR
  5. Select        — walk-forward validation, pick best per plant
  6. Simulate      — Monte Carlo scenario fan (best/base/worst)
  7. Fleet         — joblib-parallelised across all plants

SECTIONS
─────────────────────────────────────────────────────────────────────
  1.  Imports & constants
  2.  Data merger
  3.  STL decomposition
  4.  Future feature builder
  5.  Models: LightGBM, XGBoost, Ridge, SVR
  6.  Walk-forward validator
  7.  Model selector
  8.  Monte Carlo simulator
  9.  Fleet runner (parallelised)
  10. Entry point & usage examples
"""

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════

import warnings
import traceback
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[WARN] lightgbm not installed — pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed — pip install xgboost")

OUT_DIR  = Path("data/multivariate")
HORIZON  = 72      # hours ahead
VAL_DAYS = 14      # walk-forward validation window (days)

TARGET = "generation"   # ← the column we forecast

# ── EXOGENOUS features ─────────────────────────────────────────────
# These are either known for the future window (time-derived, static plant
# metadata) or come directly from forecast_df columns.  No historical
# generation needed to compute them.
EXOGENOUS_FEATURES = [
    # ── Weather (from forecast_df or carry-forward) ──────────────
    "irradiance_adjusted",
    "irradiance_ratio",
    "clear_sky_irradiance",
    "temp_effect",
    "cloud_cover",
    "wind_speed",
    "wind_direction",
    "temperature",
    # ── Calendar — always known in the future ────────────────────
    "hour", "day_of_year", "month", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "month_sin", "month_cos",
    # ── Plant type — static flags ─────────────────────────────────
    "plant_type_code",
    "plant_type_Hybrid",
    "plant_type_Solar",
    "plant_type_Wind",
    # ── Plant health — static or slow-changing ────────────────────
    "capacity_mw",
    "is_degraded",
    "is_offline",
    "health_factor",                # carry-forward or health_fc_df
    "health_adjusted_capacity_mw",  # = capacity_mw * health_factor
    # ── Maintenance state — carry-forward ─────────────────────────
    "days_since_cleaning",
    "soiling_loss",
    # ── Operational assumptions for future window ─────────────────
    "curtailment_mw",               # assume 0 in forecast window
    "availability_mw",              # carry-forward or assume = capacity
    "net_availability_mw",          # carry-forward
    # ── Derived availability flags ────────────────────────────────
    "is_daylight",                  # computed from hour
    "is_peak",                      # from prior 168h rolling max proxy
]

# ── ENDOGENOUS features ────────────────────────────────────────────
# These depend on historical generation values.  For the forecast window
# they are built using the univariate forecast as a generation proxy.
# All of these exist in both historical_df and forecast_df already.
ENDOGENOUS_FEATURES = [
    # ── Lag features ──────────────────────────────────────────────
    "gen_lag_1",
    "gen_lag_24",
    "gen_lag_168",
    # ── Rolling statistics ────────────────────────────────────────
    "gen_rolling_mean_6",
    "gen_rolling_std_6",
    "gen_rolling_mean_24",
    "gen_rolling_mean_168",
    # ── Performance ratio rolling ─────────────────────────────────
    "pr_rolling_7",
    "pr_rolling_30",
    # ── Derived generation metrics ────────────────────────────────
    "cuf",
    "load_factor",
    "expected_generation",
    "performance_ratio",
    "generation_norm",
    "adjusted_generation_signal",
    "capacity_factor",
    "generation_shortfall_mw",
    # ── Momentum & ramp ───────────────────────────────────────────
    "ramp_rate",
    "ramp_abs",
    "gen_momentum_3",
    "gen_momentum_6",
    # ── Variability ───────────────────────────────────────────────
    "gen_variability_24",
    "gen_variability_168",
    # ── Residual & normalised ─────────────────────────────────────
    "gen_residual_24",
    "gen_normalized",
    # ── Zero-generation flags ─────────────────────────────────────
    "is_zero_gen",
    "zero_streak",
    # ── Daily aggregate ───────────────────────────────────────────
    "daily_generation",
]


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA MERGER
# Stitch historical_df + forecast_df (which already has all feature
# columns populated for t+1..t+72) into one extended dataframe.
# ══════════════════════════════════════════════════════════════════════

def merge_for_multivariate(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    weather_future_df: pd.DataFrame | None = None,
    irradiance_fc_df: pd.DataFrame | None = None,
    health_fc_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge historical and forecast DataFrames into one extended DataFrame.

    Both DataFrames already share the same column schema (all feature
    columns present in both).  This function:
      1. Tags rows with is_forecast flag
      2. Optionally enriches future weather / irradiance / health
         columns from auxiliary forecast DataFrames
      3. Recomputes any endogenous features that depend on the full
         extended series (lags that cross the history/forecast boundary)
      4. Returns one DataFrame per plant concatenated together

    Parameters
    ----------
    historical_df     : Output of build_features() — past rows.
                        Must have: plant_id, timestamp, generation,
                                   + all feature columns.
    forecast_df       : Pre-built future-window DataFrame.
                        Same schema as historical_df.
                        generation column = univariate forecast_mw proxy.
                        Must have: plant_id, timestamp, forecast_mw,
                                   lower_90, upper_90.
    weather_future_df : Richer future weather (optional).
                        Columns: timestamp, [plant_id or region],
                                 irradiance_wm2, cloud_cover_pct,
                                 wind_speed_kmh, wind_direction_deg.
                        When provided, overrides weather columns in
                        forecast_df for matched timestamps.
    irradiance_fc_df  : Irradiance forecast from auxiliary_forecasts.py
                        (optional). Columns: plant_id, timestamp,
                        irradiance_forecast.
    health_fc_df      : Health forecast from auxiliary_forecasts.py
                        (optional). Columns: plant_id, timestamp,
                        health_forecast, [repair_probability].

    Returns
    -------
    pd.DataFrame : Combined historical + forecast rows with is_forecast
                   flag and all feature columns populated.
    """
    # ── Standardise timestamps & add flags ────────────────────────
    historical_df = historical_df.copy()
    historical_df["timestamp"]   = pd.to_datetime(historical_df["timestamp"])
    historical_df["is_forecast"] = False

    forecast_df = forecast_df.copy()
    forecast_df["timestamp"]   = pd.to_datetime(forecast_df["timestamp"])
    forecast_df["is_forecast"] = True

    print(f"[Merge] Historical: {len(historical_df):,} rows  |  "
          f"Forecast: {len(forecast_df):,} rows  |  "
          f"Plants: {historical_df['plant_id'].nunique()}")

    # ── Optionally overwrite weather columns in forecast_df ───────
    if weather_future_df is not None:
        forecast_df = _apply_future_weather(forecast_df, weather_future_df)

    # ── Optionally overwrite irradiance from auxiliary forecast ───
    if irradiance_fc_df is not None:
        irr = irradiance_fc_df[["plant_id", "timestamp", "irradiance_forecast"]].copy()
        irr["timestamp"] = pd.to_datetime(irr["timestamp"])
        forecast_df = forecast_df.merge(irr, on=["plant_id", "timestamp"], how="left")
        matched = forecast_df["irradiance_forecast"].notna()
        forecast_df.loc[matched, "irradiance_adjusted"] = forecast_df.loc[matched, "irradiance_forecast"]
        forecast_df.drop(columns=["irradiance_forecast"], errors="ignore", inplace=True)
        print(f"[Merge]   irradiance_fc applied to {matched.sum():,} rows")

    # ── Optionally overwrite health from auxiliary forecast ───────
    if health_fc_df is not None:
        hlt_cols = ["plant_id", "timestamp", "health_forecast"]
        if "repair_probability" in health_fc_df.columns:
            hlt_cols.append("repair_probability")
        hlt = health_fc_df[hlt_cols].copy()
        hlt["timestamp"] = pd.to_datetime(hlt["timestamp"])
        forecast_df = forecast_df.merge(hlt, on=["plant_id", "timestamp"], how="left")
        matched = forecast_df["health_forecast"].notna()
        forecast_df.loc[matched, "health_factor"] = forecast_df.loc[matched, "health_forecast"]
        # Update health_adjusted_capacity_mw
        forecast_df["health_adjusted_capacity_mw"] = (
            forecast_df["capacity_mw"] * forecast_df["health_factor"]
        )
        forecast_df.drop(columns=["health_forecast"], errors="ignore", inplace=True)
        print(f"[Merge]   health_fc applied to {matched.sum():,} rows")

    # ── Concatenate per plant and recompute boundary-crossing lags ─
    extended_parts = []
    for plant_id in historical_df["plant_id"].unique():
        hist = historical_df[historical_df["plant_id"] == plant_id].copy()
        fut  = forecast_df[forecast_df["plant_id"] == plant_id].copy()

        if len(fut) == 0:
            print(f"  [WARN] No forecast rows for {plant_id} — using history only")
            extended_parts.append(hist)
            continue

        combined = pd.concat([hist, fut], ignore_index=True).sort_values("timestamp")
        combined = _recompute_boundary_lags(combined)
        extended_parts.append(combined)

    extended = pd.concat(extended_parts, ignore_index=True)
    n_hist   = (~extended["is_forecast"]).sum()
    n_fut    = extended["is_forecast"].sum()
    print(f"[Merge] Extended df ready: {n_hist:,} historical + {n_fut:,} forecast rows")
    return extended


def _apply_future_weather(
    forecast_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Overwrite weather columns in forecast_df with richer API data.
    Matches on timestamp; falls back to plant_id match if column present.
    """
    weather_df = weather_df.copy()
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

    col_map = {
        "irradiance_wm2":     "irradiance",
        "cloud_cover_pct":    "cloud_cover",
        "wind_speed_kmh":     "wind_speed",
        "wind_direction_deg": "wind_direction",
    }

    for _, w_row in weather_df.iterrows():
        ts = w_row["timestamp"]
        mask = forecast_df["timestamp"] == ts
        if "plant_id" in weather_df.columns:
            mask = mask & (forecast_df["plant_id"] == w_row["plant_id"])
        for src_col, tgt_col in col_map.items():
            if src_col in w_row.index and tgt_col in forecast_df.columns:
                forecast_df.loc[mask, tgt_col] = w_row[src_col]
        # Recompute irradiance_adjusted from raw irradiance + cloud cover
        if "irradiance" in forecast_df.columns and "cloud_cover" in forecast_df.columns:
            forecast_df.loc[mask, "irradiance_adjusted"] = (
                forecast_df.loc[mask, "irradiance"]
                * (1 - forecast_df.loc[mask, "cloud_cover"] / 100)
            ).clip(lower=0)

    n_matched = (forecast_df["timestamp"].isin(weather_df["timestamp"])).sum()
    print(f"[Merge]   weather_future applied to ~{n_matched} rows")
    return forecast_df


def _recompute_boundary_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute lag/rolling features that cross the history/forecast
    boundary.  Only updates NaN slots — pre-computed historical values
    are preserved.
    gen_lag_1, gen_lag_24, gen_lag_168: recomputed on full series.
    gen_rolling_mean_6/24/168, gen_rolling_std_6: recomputed.
    Momentum and ramp columns: recomputed.
    All other endogenous columns already present in forecast_df
    (built by the preprocessing pipeline) — not touched.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    g  = df["generation"]

    # Only fill NaN cells — preserves clean historical values
    def _fill(col_name: pd.Series, values: pd.Series):
        if col_name not in df.columns:
            df[col_name] = values
        else:
            df[col_name] = df[col_name].combine_first(values)

    _fill("gen_lag_1",           g.shift(1))
    _fill("gen_lag_24",          g.shift(24))
    _fill("gen_lag_168",         g.shift(168))
    _fill("gen_rolling_mean_6",  g.shift(1).rolling(6,   min_periods=1).mean())
    _fill("gen_rolling_std_6",   g.shift(1).rolling(6,   min_periods=2).std().fillna(0))
    _fill("gen_rolling_mean_24", g.shift(1).rolling(24,  min_periods=1).mean())
    _fill("gen_rolling_mean_168",g.shift(1).rolling(168, min_periods=1).mean())
    _fill("ramp_rate",           g.diff())
    _fill("ramp_abs",            g.diff().abs())
    _fill("gen_momentum_3",      g - g.shift(3))
    _fill("gen_momentum_6",      g - g.shift(6))
    _fill("gen_residual_24",     g - g.shift(1).rolling(24, min_periods=1).mean())
    rolling168_mean = g.shift(1).rolling(168, min_periods=1).mean()
    _fill("gen_normalized",      g / (rolling168_mean + 1e-6))

    # is_zero_gen / zero_streak — forward-propagate
    if "is_zero_gen" in df.columns:
        is_zero_new = (g < 0.01).astype(int)
        df["is_zero_gen"] = df["is_zero_gen"].combine_first(is_zero_new)

    return df


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — STL DECOMPOSITION
# Understand trend + seasonality + residual before modelling.
# ══════════════════════════════════════════════════════════════════════

def decompose_series(
    series: pd.Series,
    plant_id: str,
    period: int = 24,       # daily seasonality (hourly data)
    seasonal: int = 7,      # STL seasonal smoother window (must be odd)
    weekly_period: int = 168,  # also check weekly seasonality
) -> dict:
    """
    STL (Seasonal-Trend decomposition using LOESS) on the generation series.

    Why STL over classical decomposition:
      - Robust to outliers via LOESS local regression
      - Handles solar zero-at-night series cleanly
      - Residual exposes unexplained variance:
          high residual_std → hard to forecast, consider more features
      - Two decompositions run:
          1. Daily   (period=24)  — captures intraday solar curve
          2. Weekly  (period=168) — captures weekly dispatch pattern

    Returns
    -------
    dict:
      plant_id, stl_daily, stl_weekly (STL result objects),
      trend, seasonal, residual (from daily decomp),
      seasonal_strength, trend_strength, residual_std, residual_mean,
      dominant_period ("daily" | "weekly"),
      forecast_difficulty ("easy" | "medium" | "hard")
    """
    series = series.dropna()
    if len(series) < period * 2:
        raise ValueError(
            f"[{plant_id}] Series too short for STL: "
            f"need {period*2}, got {len(series)}"
        )

    def _run_stl(p: int) -> STL:
        s = max(7, p // 3 | 1)   # odd seasonal smoother ≥ 7
        return STL(
            series,
            period=p,
            seasonal=s,
            trend=None,
            robust=True,
        ).fit()

    # ── Daily decomposition (always run) ──────────────────────────
    result_daily = _run_stl(period)
    trend    = result_daily.trend
    seasonal = result_daily.seasonal
    residual = result_daily.resid

    var_r  = residual.var()
    ss_daily = max(0, 1 - var_r / (var_r + seasonal.var() + 1e-9))
    ts_daily = max(0, 1 - var_r / (var_r + trend.var()   + 1e-9))

    # ── Weekly decomposition (if enough data) ─────────────────────
    result_weekly = None
    ss_weekly = 0.0
    if len(series) >= weekly_period * 2:
        result_weekly = _run_stl(weekly_period)
        r_w  = result_weekly.resid
        ss_weekly = max(0, 1 - r_w.var() / (
            r_w.var() + result_weekly.seasonal.var() + 1e-9
        ))

    dominant_period = "weekly" if ss_weekly > ss_daily else "daily"

    print(f"[{plant_id}] STL decomposition:")
    print(f"  Daily seasonal strength  : {ss_daily:.3f}  "
          f"({'strong' if ss_daily > 0.6 else 'moderate' if ss_daily > 0.3 else 'weak'})")
    if result_weekly:
        print(f"  Weekly seasonal strength : {ss_weekly:.3f}  "
              f"({'strong' if ss_weekly > 0.6 else 'moderate' if ss_weekly > 0.3 else 'weak'})")
    print(f"  Trend strength           : {ts_daily:.3f}")
    print(f"  Residual std             : {residual.std():.3f} MW  (unexplained noise)")
    print(f"  Dominant period          : {dominant_period}")

    decomp = {
        "plant_id":          plant_id,
        "trend":             trend,
        "seasonal":          seasonal,
        "residual":          residual,
        "seasonal_strength": round(float(ss_daily),    4),
        "trend_strength":    round(float(ts_daily),    4),
        "weekly_strength":   round(float(ss_weekly),   4),
        "residual_std":      round(float(residual.std()), 4),
        "residual_mean":     round(float(residual.mean()), 4),
        "dominant_period":   dominant_period,
        "stl_daily":         result_daily,
        "stl_weekly":        result_weekly,
    }
    decomp["forecast_difficulty"] = _forecast_difficulty(decomp)
    return decomp


def decompose_fleet(
    feature_df: pd.DataFrame,
    period: int = 24,
) -> pd.DataFrame:
    """
    Run STL decomposition for all plants and return a summary DataFrame.

    Use this BEFORE training to:
      - Identify which plants are easy vs hard to forecast
      - Spot plants with unusual residual behaviour
      - Guide feature engineering decisions

    Returns
    -------
    pd.DataFrame sorted by residual_std descending:
      plant_id, seasonal_strength, weekly_strength, trend_strength,
      residual_std, dominant_period, forecast_difficulty
    """
    records = []
    for plant_id, plant_df in feature_df.groupby("plant_id"):
        series = (
            plant_df.sort_values("timestamp")
            .set_index("timestamp")[TARGET]
            .asfreq("h")
            .fillna(0)
        )
        try:
            d = decompose_series(series, plant_id, period=period)
            records.append({
                "plant_id":            plant_id,
                "seasonal_strength":   d["seasonal_strength"],
                "weekly_strength":     d["weekly_strength"],
                "trend_strength":      d["trend_strength"],
                "residual_std":        d["residual_std"],
                "dominant_period":     d["dominant_period"],
                "forecast_difficulty": d["forecast_difficulty"],
            })
        except Exception as e:
            print(f"  [WARN] STL failed for {plant_id}: {e}")

    summary = (
        pd.DataFrame(records)
        .sort_values("residual_std", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n[Decompose] Fleet STL summary ({len(summary)} plants):")
    print(summary.to_string(index=False))
    _print_difficulty_breakdown(summary)
    return summary


def _forecast_difficulty(decomp: dict) -> str:
    """Classify forecast difficulty from STL decomposition metrics."""
    ss = decomp["seasonal_strength"]
    ts = decomp["trend_strength"]
    rs = decomp["residual_std"]
    if ss > 0.7 and rs < 5:
        return "easy"
    elif ss > 0.4 and rs < 15:
        return "medium"
    else:
        return "hard"


def _print_difficulty_breakdown(summary: pd.DataFrame) -> None:
    for level in ["easy", "medium", "hard"]:
        n = (summary["forecast_difficulty"] == level).sum()
        pct = n / len(summary) * 100 if len(summary) else 0
        print(f"  {level:6s}: {n} plants ({pct:.0f}%)")


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    extended_df: pd.DataFrame,
    plant_id: str,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Split extended DataFrame (historical + forecast rows) into:
      train_df  : historical rows (is_forecast == False)
      future_df : forecast rows  (is_forecast == True)

    feature_cols defaults to EXOGENOUS_FEATURES + ENDOGENOUS_FEATURES,
    filtered to columns that actually exist in the DataFrame.

    Rows with any NaN in the selected features are dropped from train_df
    only; future_df NaNs are forward-filled so prediction can proceed.

    Returns
    -------
    (train_df, future_df, resolved_feature_cols)
    """
    if feature_cols is None:
        feature_cols = [
            c for c in EXOGENOUS_FEATURES + ENDOGENOUS_FEATURES
            if c in extended_df.columns
        ]

    plant_df = (
        extended_df[extended_df["plant_id"] == plant_id]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    train_df  = plant_df[~plant_df["is_forecast"]].copy()
    future_df = plant_df[ plant_df["is_forecast"]].copy()

    # Training: drop rows with any NaN feature (keeps model clean)
    before = len(train_df)
    train_df = train_df.dropna(subset=feature_cols).reset_index(drop=True)
    dropped  = before - len(train_df)
    if dropped:
        print(f"  [{plant_id}] Dropped {dropped} training rows with NaN features")

    # Future: forward-fill NaNs (can't drop — need all 72 rows)
    future_df[feature_cols] = (
        future_df[feature_cols].ffill().bfill().fillna(0)
    )

    print(f"[{plant_id}] Feature matrix: "
          f"train={len(train_df):,}  future={len(future_df):,}  "
          f"features={len(feature_cols)}")

    return train_df, future_df, feature_cols


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — MODELS
# ══════════════════════════════════════════════════════════════════════

def _evaluate(actual: np.ndarray, predicted: np.ndarray, name: str) -> dict:
    """Compute MAE, RMSE, MAPE. Clips predictions to >= 0."""
    n    = min(len(actual), len(predicted))
    act  = actual[:n]
    pred = np.maximum(0, predicted[:n])
    mae  = mean_absolute_error(act, pred)
    rmse = np.sqrt(mean_squared_error(act, pred))
    mask = act > 0.1
    mape = (
        float(np.mean(np.abs((act[mask] - pred[mask]) / (act[mask] + 1e-6))) * 100)
        if mask.sum() else np.nan
    )
    print(f"    {name:12s} → MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%")
    return {"model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "MAPE": round(mape, 2)}


# ── LightGBM ──────────────────────────────────────────────────────────

def fit_lgbm_multivariate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    feature_names: list[str],
) -> tuple[object, dict]:
    """
    LightGBM gradient-boosted trees.

    All future feature values are pre-computed and live in X_future,
    so no recursive step is needed — this is a pure tabular regression.

    Hyperparameters tuned for:
      - Hourly renewable energy forecasting (high seasonality)
      - 72-step horizon (requires generalisation, not overfitting)
      - ~50 plants × years of history (medium-large dataset)
    """
    if not HAS_LGBM:
        raise ImportError("lightgbm not installed — pip install lightgbm")

    model = lgb.LGBMRegressor(
        n_estimators=1500,
        num_leaves=63,
        learning_rate=0.025,
        feature_fraction=0.75,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
        n_jobs=1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        feature_name=feature_names,
        callbacks=[
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    val_pred = model.predict(X_val)
    metrics  = _evaluate(y_val, val_pred, "lightgbm")

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)
    print(f"    Top-5 features: {importance.head(5).index.tolist()}")

    return model, metrics


# ── XGBoost ───────────────────────────────────────────────────────────

def fit_xgb_multivariate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> tuple[object, dict]:
    """
    XGBoost gradient-boosted trees.
    Stronger regularisation than LightGBM; often better on smaller plants.
    """
    if not HAS_XGB:
        raise ImportError("xgboost not installed — pip install xgboost")

    model = xgb.XGBRegressor(
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        verbosity=0,
        n_jobs=1,
        early_stopping_rounds=150,
        eval_metric="mae",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    val_pred = model.predict(X_val)
    metrics  = _evaluate(y_val, val_pred, "xgboost")
    return model, metrics


# ── Ridge Regression ──────────────────────────────────────────────────

def fit_ridge_multivariate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> tuple[object, dict]:
    """
    Ridge regression with StandardScaler.

    Fast, interpretable, excellent baseline.
    Works particularly well when irradiance → generation is near-linear
    (common for well-maintained solar plants with low soiling loss).
    Usually wins on plants classified "easy" by STL decomposition.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])
    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    metrics  = _evaluate(y_val, val_pred, "ridge")
    return pipe, metrics


# ── SVR ───────────────────────────────────────────────────────────────

def fit_svr_multivariate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> tuple[object, dict]:
    """
    Support Vector Regression with RBF kernel.

    Captures non-linear irradiance-to-generation curves (e.g. wind
    plants with cut-in / rated / cut-out speed regions).
    Capped at 8,000 training rows because SVR is O(n²) — on large
    plants it samples randomly rather than failing.
    """
    n_fit = min(len(X_train), 8_000)
    if n_fit < len(X_train):
        print(f"    SVR: sampling {n_fit} / {len(X_train)} train rows (O(n²) cap)")
        idx    = np.random.choice(len(X_train), n_fit, replace=False)
        X_fit  = X_train[idx]
        y_fit  = y_train[idx]
    else:
        X_fit, y_fit = X_train, y_train

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", C=10, epsilon=0.5, gamma="scale")),
    ])
    pipe.fit(X_fit, y_fit)
    val_pred = pipe.predict(X_val)
    metrics  = _evaluate(y_val, val_pred, "svr")
    return pipe, metrics


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — WALK-FORWARD VALIDATOR
# ══════════════════════════════════════════════════════════════════════

def walk_forward_validate(
    train_df:     pd.DataFrame,
    feature_cols: list[str],
    val_days:     int = VAL_DAYS,
    n_splits:     int = 3,
) -> dict[str, list[dict]]:
    """
    Walk-forward (expanding window) cross-validation.

    Why walk-forward instead of k-fold:
      - k-fold uses future rows to predict the past → data leakage
      - Walk-forward mirrors production: train on all past, predict next window
      - n_splits=3 gives 3 successive val windows, each val_days days long

    Split logic:
      Fold 1: train=[0..N-3*val_h],  val=[N-3*val_h .. N-2*val_h]
      Fold 2: train=[0..N-2*val_h],  val=[N-2*val_h .. N-1*val_h]
      Fold 3: train=[0..N-1*val_h],  val=[N-1*val_h .. N]

    Returns
    -------
    {model_name: [metrics_fold_1, metrics_fold_2, ...]}
    """
    df       = train_df.sort_values("timestamp").reset_index(drop=True)
    val_size = val_days * 24    # hours
    n        = len(df)
    min_train = max(val_size * 2, 500)

    if n < min_train + val_size:
        raise ValueError(
            f"Insufficient data for walk-forward: "
            f"need {min_train + val_size} rows, got {n}"
        )

    all_scores: dict[str, list] = {}

    for fold in range(n_splits):
        val_end   = n - fold * val_size
        val_start = val_end - val_size
        if val_start < min_train:
            print(f"  [Fold {fold+1}] Skipped — insufficient training rows")
            continue

        tr = df.iloc[:val_start]
        vl = df.iloc[val_start:val_end]

        X_tr, y_tr = tr[feature_cols].values, tr[TARGET].values
        X_vl, y_vl = vl[feature_cols].values, vl[TARGET].values

        print(f"\n  Fold {fold+1}/{n_splits}: "
              f"train={len(tr):,} rows  val={len(vl):,} rows")

        for name, fn in [
            ("lightgbm", lambda: fit_lgbm_multivariate(X_tr, y_tr, X_vl, y_vl, feature_cols) if HAS_LGBM else None),
            ("xgboost",  lambda: fit_xgb_multivariate( X_tr, y_tr, X_vl, y_vl) if HAS_XGB else None),
            ("ridge",    lambda: fit_ridge_multivariate(X_tr, y_tr, X_vl, y_vl)),
            ("svr",      lambda: fit_svr_multivariate(  X_tr, y_tr, X_vl, y_vl)),
        ]:
            try:
                out = fn()
                if out is None:
                    continue
                _, m = out
                all_scores.setdefault(name, []).append(m)
            except Exception as e:
                print(f"    {name} fold {fold+1} failed: {e}")

    return all_scores


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — MODEL SELECTOR (per-plant full pipeline)
# ══════════════════════════════════════════════════════════════════════

def select_best_multivariate_model(
    extended_df:   pd.DataFrame,
    plant_id:      str,
    feature_cols:  list[str] | None = None,
    val_days:      int = VAL_DAYS,
    n_splits:      int = 3,
) -> dict:
    """
    Full end-to-end pipeline for a single plant:

      1. Build feature matrix (train / future split)
      2. STL decomposition (diagnostic)
      3. Walk-forward cross-validation across all models
      4. Select best model (lowest average MAE)
      5. Refit best model on full training data
      6. Predict on t+1..t+72 future feature rows
      7. Build confidence interval

    Returns
    -------
    dict:
      plant_id, best_model, feature_cols, decomposition,
      validation_scores, avg_scores, final_model,
      forecast_df (plant_id, timestamp, forecast_mw, lower_90, upper_90, model)
    """
    print(f"\n{'═'*60}")
    print(f"  MULTIVARIATE: {plant_id}")
    print(f"{'═'*60}")

    # ── Feature matrix ─────────────────────────────────────────────
    train_df, future_df, feat_cols = build_feature_matrix(
        extended_df, plant_id, feature_cols
    )
    if len(train_df) < 500:
        raise ValueError(f"Insufficient training rows: {len(train_df)}")
    if len(future_df) == 0:
        raise ValueError("No future rows — check forecast_df was merged")

    # ── STL decomposition (diagnostic only) ────────────────────────
    series = (
        train_df.sort_values("timestamp")
        .set_index("timestamp")[TARGET]
        .asfreq("h")
        .fillna(0)
    )
    try:
        decomp = decompose_series(series, plant_id)
    except Exception as e:
        print(f"  [WARN] STL failed: {e}")
        decomp = None

    # ── Walk-forward validation ────────────────────────────────────
    print(f"\n  Walk-forward validation ({n_splits} folds × {val_days} days):")
    fold_scores = walk_forward_validate(train_df, feat_cols, val_days, n_splits)

    if not fold_scores:
        raise RuntimeError(f"All models failed for {plant_id}")

    avg_scores = {
        model: {
            "MAE":  round(np.mean([f["MAE"]  for f in folds]), 3),
            "RMSE": round(np.mean([f["RMSE"] for f in folds]), 3),
            "MAPE": round(np.nanmean([f["MAPE"] for f in folds]), 2),
        }
        for model, folds in fold_scores.items()
    }

    best_name = min(avg_scores, key=lambda m: avg_scores[m]["MAE"])

    print(f"\n  Validation results (avg across {n_splits} folds):")
    for m, sc in sorted(avg_scores.items(), key=lambda x: x[1]["MAE"]):
        flag = " ← WINNER" if m == best_name else ""
        print(f"    {m:12s}: MAE={sc['MAE']:.3f}  "
              f"RMSE={sc['RMSE']:.3f}  MAPE={sc['MAPE']:.1f}%{flag}")

    # ── Refit best model on full training data ─────────────────────
    print(f"\n  Refitting {best_name} on full training set "
          f"({len(train_df):,} rows)...")

    X_all = train_df[feat_cols].values
    y_all = train_df[TARGET].values
    split = max(len(X_all) - val_days * 24, int(len(X_all) * 0.85))
    X_tr, X_vl = X_all[:split], X_all[split:]
    y_tr, y_vl = y_all[:split], y_all[split:]

    fitters = {
        "lightgbm": lambda: fit_lgbm_multivariate(X_tr, y_tr, X_vl, y_vl, feat_cols),
        "xgboost":  lambda: fit_xgb_multivariate( X_tr, y_tr, X_vl, y_vl),
        "ridge":    lambda: fit_ridge_multivariate(X_tr, y_tr, X_vl, y_vl),
        "svr":      lambda: fit_svr_multivariate(  X_tr, y_tr, X_vl, y_vl),
    }
    final_model, _ = fitters[best_name]()

    # ── Predict on future rows ─────────────────────────────────────
    X_future = future_df[feat_cols].values
    fc_mw    = np.maximum(0, final_model.predict(X_future))

    # Confidence interval: ±1.5 × validation RMSE (simple but calibrated)
    val_rmse = avg_scores[best_name]["RMSE"]
    fc_lower = np.maximum(0, fc_mw - 1.5 * val_rmse)
    fc_upper = fc_mw + 1.5 * val_rmse

    # Blend with univariate CI where available
    if "lower_90" in future_df.columns and "upper_90" in future_df.columns:
        u_lo = future_df["lower_90"].values
        u_hi = future_df["upper_90"].values
        fc_lower = np.minimum(fc_lower, u_lo)
        fc_upper = np.maximum(fc_upper, u_hi)
        # Trust multivariate prediction — allow its own ±15% headroom
        fc_lower = np.minimum(fc_lower, fc_mw * 0.85)
        fc_upper = np.maximum(fc_upper, fc_mw * 1.15)

    forecast_df_out = pd.DataFrame({
        "plant_id":    plant_id,
        "timestamp":   future_df["timestamp"].values,
        "forecast_mw": np.round(fc_mw,    3),
        "lower_90":    np.round(fc_lower, 3),
        "upper_90":    np.round(fc_upper, 3),
        "model":       best_name,
    })

    print(f"\n  Forecast ready: {len(forecast_df_out)} hours | "
          f"range [{fc_mw.min():.1f}, {fc_mw.max():.1f}] MW")

    return {
        "plant_id":          plant_id,
        "best_model":        best_name,
        "feature_cols":      feat_cols,
        "decomposition":     decomp,
        "validation_scores": fold_scores,
        "avg_scores":        avg_scores,
        "final_model":       final_model,
        "forecast_df":       forecast_df_out,
    }


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — MONTE CARLO SCENARIO SIMULATOR
# ══════════════════════════════════════════════════════════════════════

def simulate_scenarios(
    forecast_df:   pd.DataFrame,
    decomp:        dict | None,
    capacity_mw:   float | None = None,
    n_simulations: int = 1000,
    seed:          int = 42,
) -> pd.DataFrame:
    """
    Monte Carlo simulation producing a probability fan around the
    point forecast.

    Three named scenarios:
      P10 (worst case)  : 10th percentile of 1000 simulated paths
      P50 (base case)   : 50th percentile (median)
      P90 (best case)   : 90th percentile

    Four independent uncertainty sources (combined in quadrature):
      1. Model uncertainty      : from CI width in forecast_df
                                  (= walkforward RMSE × 1.5)
      2. Weather uncertainty    : 8% of fc (irradiance forecast error)
      3. Soiling / degradation  : 3% bias on is_degraded plants
      4. Residual noise         : N(0, residual_std) from STL decomp
                                  (captures unexplained variance)

    Autocorrelation: consecutive forecast hours are correlated (ρ=0.7)
    — cloud ramps are persistent, not random.

    Parameters
    ----------
    forecast_df   : Output of select_best_multivariate_model().
                    Required columns: plant_id, timestamp,
                                      forecast_mw, lower_90, upper_90.
    decomp        : STL decomposition dict.  Pass None → uses 10% fallback.
    capacity_mw   : Plant capacity cap (optional). Prevents simulations
                    exceeding physical maximum.  If None → 2× max forecast.
    n_simulations : Monte Carlo paths (1000 is fast and stable).
    seed          : Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame columns:
      plant_id, timestamp, forecast_mw,
      p10, p25, p50, p75, p90,
      scenario_worst, scenario_base, scenario_best,
      sigma_total
    """
    rng      = np.random.RandomState(seed)
    n_steps  = len(forecast_df)
    fc       = forecast_df["forecast_mw"].values.astype(float)

    # ── Sigma per uncertainty source ──────────────────────────────
    ci_width     = (
        forecast_df["upper_90"].values - forecast_df["lower_90"].values
    ).astype(float)
    # 90% CI = ±1.645σ  →  σ = CI_width / (2 × 1.645)
    sigma_model  = ci_width / (2 * 1.645)

    sigma_resid  = (
        decomp["residual_std"] if (decomp and "residual_std" in decomp)
        else fc.mean() * 0.10
    )
    sigma_weather = fc * 0.08    # 8% OWM irradiance forecast error
    sigma_soiling = fc * 0.03    # 3% soiling bias (small negative mean below)

    # Total σ (quadrature)
    sigma_total = np.sqrt(
        sigma_model ** 2
        + sigma_resid ** 2
        + sigma_weather ** 2
        + sigma_soiling ** 2
    )

    # ── Simulate n_simulations paths with AR(1) noise ─────────────
    noise = rng.randn(n_simulations, n_steps)
    for t in range(1, n_steps):
        noise[:, t] = 0.70 * noise[:, t - 1] + 0.30 * noise[:, t]

    # Small negative bias from soiling/degradation
    soiling_bias = fc * 0.015    # 1.5% expected generation loss
    simulations  = (
        fc[None, :]
        + sigma_total[None, :] * noise
        - soiling_bias[None, :]
    )
    simulations = np.maximum(0, simulations)

    # Capacity cap
    cap = capacity_mw if capacity_mw else fc.max() / 0.5
    simulations = np.minimum(simulations, cap)

    # ── Percentiles ───────────────────────────────────────────────
    p10 = np.percentile(simulations, 10, axis=0)
    p25 = np.percentile(simulations, 25, axis=0)
    p50 = np.percentile(simulations, 50, axis=0)
    p75 = np.percentile(simulations, 75, axis=0)
    p90 = np.percentile(simulations, 90, axis=0)

    print(f"[Simulate] {n_simulations} Monte Carlo paths | {n_steps} steps")
    print(f"  Point forecast total : {fc.sum():.1f} MWh")
    print(f"  Base (P50) total     : {p50.sum():.1f} MWh")
    print(f"  Best (P90) total     : {p90.sum():.1f} MWh  "
          f"(+{(p90.sum()-p50.sum())/max(p50.sum(),1)*100:.1f}%)")
    print(f"  Worst(P10) total     : {p10.sum():.1f} MWh  "
          f"({(p10.sum()-p50.sum())/max(p50.sum(),1)*100:.1f}%)")

    return pd.DataFrame({
        "plant_id":       forecast_df["plant_id"].values,
        "timestamp":      forecast_df["timestamp"].values,
        "forecast_mw":    fc.round(3),
        "p10":            p10.round(3),
        "p25":            p25.round(3),
        "p50":            p50.round(3),
        "p75":            p75.round(3),
        "p90":            p90.round(3),
        "scenario_worst": p10.round(3),
        "scenario_base":  p50.round(3),
        "scenario_best":  p90.round(3),
        "sigma_total":    sigma_total.round(3),
    })


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — FLEET RUNNER (parallelised)
# ══════════════════════════════════════════════════════════════════════

def _process_one_plant_mv(
    plant_id:      str,
    extended_df:   pd.DataFrame,
    feature_cols:  list[str] | None,
    val_days:      int,
    n_splits:      int,
    n_simulations: int,
) -> dict:
    """
    Worker function for joblib — processes one plant end-to-end.
    Never raises — all exceptions are captured and returned in 'error'.
    """
    warnings.filterwarnings("ignore")
    try:
        result = select_best_multivariate_model(
            extended_df, plant_id, feature_cols, val_days, n_splits
        )

        # Pass capacity_mw into simulator for realistic cap
        cap_mw = None
        if "capacity_mw" in extended_df.columns:
            cap_mw = float(
                extended_df.loc[
                    extended_df["plant_id"] == plant_id, "capacity_mw"
                ].iloc[0]
            )

        sim_df = simulate_scenarios(
            result["forecast_df"],
            result["decomposition"],
            capacity_mw=cap_mw,
            n_simulations=n_simulations,
        )

        decomp_summary = None
        if result["decomposition"]:
            d = result["decomposition"]
            decomp_summary = {
                "seasonal_strength":   d["seasonal_strength"],
                "weekly_strength":     d["weekly_strength"],
                "trend_strength":      d["trend_strength"],
                "residual_std":        d["residual_std"],
                "dominant_period":     d["dominant_period"],
                "forecast_difficulty": d["forecast_difficulty"],
            }

        return {
            "status":          "ok",
            "plant_id":        plant_id,
            "forecast_df":     result["forecast_df"],
            "simulation_df":   sim_df,
            "best_model":      result["best_model"],
            "avg_scores":      result["avg_scores"],
            "decomp_summary":  decomp_summary,
            "error":           None,
        }

    except Exception as e:
        return {
            "status":          "error",
            "plant_id":        plant_id,
            "forecast_df":     None,
            "simulation_df":   None,
            "best_model":      None,
            "avg_scores":      None,
            "decomp_summary":  None,
            "error":           f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


def run_multivariate_fleet(
    historical_df:      pd.DataFrame,
    forecast_df:        pd.DataFrame,
    irradiance_fc_df:   pd.DataFrame | None = None,
    health_fc_df:       pd.DataFrame | None = None,
    weather_future_df:  pd.DataFrame | None = None,
    feature_cols:       list[str] | None = None,
    val_days:           int  = VAL_DAYS,
    n_splits:           int  = 3,
    n_simulations:      int  = 1000,
    n_jobs:             int  = -1,
    output_dir:         str  = "data/multivariate",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full multivariate forecasting pipeline for all plants in parallel.

    Parameters
    ----------
    historical_df     : Preprocessed + feature-engineered history.
                        Required columns:
                            plant_id, timestamp, generation,
                            + all feature columns in EXOGENOUS + ENDOGENOUS lists.
                        Produced by: build_features(preprocess(raw_df))

    forecast_df       : Future-window DataFrame — same schema as historical_df.
                        Required columns:
                            plant_id, timestamp,
                            forecast_mw,    ← univariate generation proxy
                            lower_90, upper_90,
                            + all feature columns (weather, time, plant metadata)
                        Produced by: run_univariate_fleet() + preprocessing
                        NOTE: This is a DataFrame, NOT a CSV path.

    irradiance_fc_df  : Irradiance forecast from auxiliary_forecasts.py (optional).
                        Required columns:
                            plant_id, timestamp, irradiance_forecast
                        When provided, overwrites irradiance_adjusted in
                        forecast_df for matched plant+timestamp rows.

    health_fc_df      : Health forecast from auxiliary_forecasts.py (optional).
                        Required columns:
                            plant_id, timestamp, health_forecast,
                            [repair_probability]
                        When provided, overwrites health_factor and
                        health_adjusted_capacity_mw in forecast_df.

    weather_future_df : Richer future weather from OWM API (optional).
                        Required columns:
                            timestamp, [plant_id or region],
                            irradiance_wm2, cloud_cover_pct,
                            wind_speed_kmh, wind_direction_deg
                        When provided, overwrites weather columns in
                        forecast_df for matched timestamps.

    feature_cols      : Explicit feature list after any external feature
                        reduction step.  If None → auto-selects from
                        EXOGENOUS_FEATURES + ENDOGENOUS_FEATURES,
                        filtered to columns present in the DataFrame.

    val_days          : Days per walk-forward validation fold. Default: 14.
    n_splits          : Number of walk-forward folds. Default: 3.
    n_simulations     : Monte Carlo paths per plant. Default: 1000.
    n_jobs            : Parallel workers. -1 = all CPU cores.
    output_dir        : Directory for output CSVs.

    Returns
    -------
    (forecast_df_all, simulation_df_all)

    forecast_df_all columns:
        plant_id, timestamp, forecast_mw, lower_90, upper_90, model

    simulation_df_all columns:
        plant_id, timestamp, forecast_mw,
        p10, p25, p50, p75, p90,
        scenario_worst, scenario_base, scenario_best, sigma_total

    Output files (in output_dir/):
        multivariate_forecasts.csv
        scenario_simulations.csv
        model_selection_log.csv
        failed_plants.csv  (only if any plants failed)
    """
    OUT_DIR_MV = Path(output_dir)
    OUT_DIR_MV.mkdir(parents=True, exist_ok=True)

    # ── Validate inputs ───────────────────────────────────────────
    _validate_fleet_inputs(historical_df, forecast_df,
                           irradiance_fc_df, health_fc_df)

    # ── Merge historical + forecast + auxiliary inputs ─────────────
    print("\nMerging historical + forecast DataFrames...")
    extended_df = merge_for_multivariate(
        historical_df,
        forecast_df,
        weather_future_df=weather_future_df,
        irradiance_fc_df=irradiance_fc_df,
        health_fc_df=health_fc_df,
    )

    plant_ids     = extended_df["plant_id"].unique().tolist()
    n_plants      = len(plant_ids)
    max_cores     = multiprocessing.cpu_count()
    resolved_jobs = max_cores if n_jobs == -1 else min(n_jobs, max_cores)
    resolved_jobs = min(resolved_jobs, n_plants)

    print(f"\n{'═'*60}")
    print(f"  MULTIVARIATE FLEET RUN")
    print(f"  Plants      : {n_plants}")
    print(f"  Horizon     : {HORIZON}h  (t+1 to t+{HORIZON})")
    print(f"  Target      : {TARGET}")
    print(f"  Val window  : {val_days} days × {n_splits} folds")
    print(f"  MC sims     : {n_simulations} paths/plant")
    print(f"  Workers     : {resolved_jobs} / {max_cores} cores")
    print(f"  Aux inputs  : "
          f"irradiance={'✓' if irradiance_fc_df is not None else '✗ (carry-fwd)'}  "
          f"health={'✓' if health_fc_df is not None else '✗ (carry-fwd)'}  "
          f"weather={'✓' if weather_future_df is not None else '✗ (carry-fwd)'}")
    print(f"{'═'*60}\n")

    # Pre-slice per plant to avoid sending full df to each subprocess
    plant_slices = {
        pid: extended_df[extended_df["plant_id"] == pid].copy()
        for pid in plant_ids
    }

    results = Parallel(n_jobs=resolved_jobs, backend="loky", verbose=10)(
        delayed(_process_one_plant_mv)(
            pid,
            plant_slices[pid],
            feature_cols,
            val_days,
            n_splits,
            n_simulations,
        )
        for pid in plant_ids
    )

    # ── Collect results ───────────────────────────────────────────
    all_forecasts   = []
    all_simulations = []
    log_rows        = []
    failed          = []

    for res in results:
        pid = res["plant_id"]
        if res["status"] == "ok":
            all_forecasts.append(res["forecast_df"])
            all_simulations.append(res["simulation_df"])
            log_row = {
                "plant_id":   pid,
                "best_model": res["best_model"],
            }
            if res["avg_scores"]:
                for m, sc in res["avg_scores"].items():
                    log_row[f"{m}_MAE"]  = sc["MAE"]
                    log_row[f"{m}_MAPE"] = sc["MAPE"]
            if res["decomp_summary"]:
                log_row.update(res["decomp_summary"])
            log_rows.append(log_row)
            best_mae = res["avg_scores"].get(res["best_model"], {}).get("MAE", "?")
            print(f"  ✓ {pid} → {res['best_model']}  MAE={best_mae}")
        else:
            first_line = res["error"].splitlines()[0]
            print(f"  ✗ {pid} → {first_line}")
            failed.append({"plant_id": pid, "error": res["error"]})

    # ── Fleet summary ─────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  FLEET COMPLETE")
    print(f"  Successful : {len(all_forecasts)} / {n_plants}")
    print(f"  Failed     : {len(failed)}")
    log_df = pd.DataFrame(log_rows)
    if len(log_df):
        print(f"\n  Model distribution:")
        for m, cnt in log_df["best_model"].value_counts().items():
            print(f"    {m:12s}: {cnt} plants ({cnt/len(log_df):.0%})")
        if "forecast_difficulty" in log_df.columns:
            print(f"\n  Forecast difficulty:")
            for lvl, cnt in log_df["forecast_difficulty"].value_counts().items():
                print(f"    {lvl:8s}: {cnt} plants")
    print(f"{'═'*60}\n")

    # ── Save ──────────────────────────────────────────────────────
    log_df.to_csv(OUT_DIR_MV / "model_selection_log.csv", index=False)
    if failed:
        pd.DataFrame(failed).to_csv(OUT_DIR_MV / "failed_plants.csv", index=False)

    if not all_forecasts:
        print("[FATAL] No multivariate forecasts generated.")
        return pd.DataFrame(), pd.DataFrame()

    forecast_df_all   = pd.concat(all_forecasts,   ignore_index=True)
    simulation_df_all = pd.concat(all_simulations, ignore_index=True)

    forecast_df_all.to_csv(   OUT_DIR_MV / "multivariate_forecasts.csv",  index=False)
    simulation_df_all.to_csv( OUT_DIR_MV / "scenario_simulations.csv",    index=False)

    print(f"  Saved:")
    print(f"    {OUT_DIR_MV}/multivariate_forecasts.csv")
    print(f"    {OUT_DIR_MV}/scenario_simulations.csv")
    print(f"    {OUT_DIR_MV}/model_selection_log.csv")

    return forecast_df_all, simulation_df_all


# ── Validation helper ──────────────────────────────────────────────────

def _validate_fleet_inputs(
    historical_df:    pd.DataFrame,
    forecast_df:      pd.DataFrame,
    irradiance_fc_df: pd.DataFrame | None,
    health_fc_df:     pd.DataFrame | None,
) -> None:
    """
    Validate all input DataFrames before any processing starts.
    Raises ValueError with a descriptive message on the first failure found.
    """
    required_hist = {"plant_id", "timestamp", TARGET}
    missing = required_hist - set(historical_df.columns)
    if missing:
        raise ValueError(
            f"historical_df missing required columns: {missing}\n"
            f"  Run preprocess() + build_features() first."
        )

    required_fc = {"plant_id", "timestamp", "forecast_mw", "lower_90", "upper_90"}
    missing = required_fc - set(forecast_df.columns)
    if missing:
        raise ValueError(
            f"forecast_df missing required columns: {missing}\n"
            f"  Output of run_univariate_fleet() has these columns."
        )

    hist_plants = set(historical_df["plant_id"].unique())
    fc_plants   = set(forecast_df["plant_id"].unique())
    no_history  = fc_plants - hist_plants
    no_forecast = hist_plants - fc_plants
    if no_history:
        print(f"  [WARN] {len(no_history)} plants in forecast_df "
              f"have no historical data: {no_history}")
    if no_forecast:
        print(f"  [WARN] {len(no_forecast)} plants in historical_df "
              f"have no forecast rows: {no_forecast}")

    if irradiance_fc_df is not None:
        req = {"plant_id", "timestamp", "irradiance_forecast"}
        missing = req - set(irradiance_fc_df.columns)
        if missing:
            raise ValueError(
                f"irradiance_fc_df missing required columns: {missing}"
            )

    if health_fc_df is not None:
        req = {"plant_id", "timestamp", "health_forecast"}
        missing = req - set(health_fc_df.columns)
        if missing:
            raise ValueError(
                f"health_fc_df missing required columns: {missing}"
            )

    print(f"[Validate] Inputs OK — "
          f"{historical_df['plant_id'].nunique()} historical plants  |  "
          f"{forecast_df['plant_id'].nunique()} forecast plants")


# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — ENTRY POINT & USAGE EXAMPLES
# ══════════════════════════════════════════════════════════════════════

# ── Convenience wrappers ────────────────────────────────────────────

def run_single_plant(
    historical_df: pd.DataFrame,
    forecast_df:   pd.DataFrame,
    plant_id:      str,
    n_simulations: int = 1000,
    irradiance_fc_df: pd.DataFrame | None = None,
    health_fc_df:     pd.DataFrame | None = None,
    weather_future_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: run the full pipeline for a single plant.

    Usage
    -----
    >>> fc_df, sim_df = run_single_plant(
    ...     historical_df  = historical_df_solar,
    ...     forecast_df    = forecast_df_solar,
    ...     plant_id       = "KA_SOLAR_001",
    ...     n_simulations  = 500,
    ... )
    >>> print(fc_df[["timestamp", "forecast_mw", "lower_90", "upper_90"]])

    Returns
    -------
    (forecast_df, simulation_df)  — for the requested plant only
    """
    extended = merge_for_multivariate(
        historical_df, forecast_df,
        weather_future_df=weather_future_df,
        irradiance_fc_df=irradiance_fc_df,
        health_fc_df=health_fc_df,
    )
    result = select_best_multivariate_model(extended, plant_id)

    cap_mw = None
    if "capacity_mw" in historical_df.columns:
        cap_mw = float(
            historical_df.loc[
                historical_df["plant_id"] == plant_id, "capacity_mw"
            ].iloc[0]
        )

    sim_df = simulate_scenarios(
        result["forecast_df"],
        result["decomposition"],
        capacity_mw=cap_mw,
        n_simulations=n_simulations,
    )
    return result["forecast_df"], sim_df


def run_decomposition_only(
    historical_df: pd.DataFrame,
    output_csv:    str = "data/multivariate/stl_fleet_summary.csv",
) -> pd.DataFrame:
    """
    Run STL decomposition fleet-wide WITHOUT training any models.
    Use before training to understand plant difficulty distribution.

    Usage
    -----
    >>> stl_summary = run_decomposition_only(historical_df_solar)
    >>> hard_plants = stl_summary[stl_summary["forecast_difficulty"] == "hard"]
    >>> print(hard_plants)
    """
    summary = decompose_fleet(historical_df)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(f"[Decompose] Saved: {output_csv}")
    return summary
