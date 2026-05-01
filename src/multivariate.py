"""
multivariate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Karnataka Renewable Energy Grid — Multivariate Forecasting Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATA FLOW
─────────────────────────────────────────────────────────────────────
HISTORICAL (known):
  feature_df     → all 32 features + generation (actual history)
  univariate_csv → forecast_mw for t+1..t+72 (fills generation gap)
  weather_df     → irradiance, wind, temp, cloud for t+1..t+72
                   (from OpenWeatherMap or synthetic future)

WHAT THIS FILE DOES:
  1. Merge         — stitch historical features + univariate forecasts
                     + future weather into one extended dataframe
  2. Decompose     — STL decomposition: trend + seasonal + residual
                     (understand what drives generation before modelling)
  3. Build future  — construct feature rows for t+1..t+72
                     using univariate forecast as "generation" proxy
                     and weather forecast as exogenous inputs
  4. Model         — LightGBM, XGBoost, Ridge, SVR multivariate models
  5. Select        — walk-forward validation, pick best per plant
  6. Simulate      — Monte Carlo scenario simulator (best/base/worst)
  7. Fleet         — joblib-parallelised across all 50 plants

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
  10. Entry point
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

# Features from your preprocessed dataframe that are EXOGENOUS
# (available for future window without needing to forecast them)
EXOGENOUS_FEATURES = [
    # Weather — available from OWM API for 48h, extended synthetically
    "irradiance_adjusted",
    "cloud_cover",
    "wind_speed",
    "wind_direction",
    # Time features — always known in future
    "hour", "day_of_year", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    # Plant metadata — static, always known
    "capacity_mw", "plant_type_code",
    "plant_type_Hybrid", "plant_type_Wind",
    "is_degraded", "is_offline",
    # Machine state — carry-forward last known value
    "days_since_cleaning",
    "curtailment_mw",       # assume 0 curtailment in forecast window
]

# Features that USE historical generation — built from univariate forecast
ENDOGENOUS_FEATURES = [
    "gen_lag_24", "gen_lag_168",
    "cuf", "expected_generation", "performance_ratio",
    "ramp_rate", "ramp_abs",
    "gen_momentum_3", "gen_momentum_6",
    "is_peak", "gen_residual_24", "gen_normalized",
]

TARGET = "generation"


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA MERGER
# Stitch historical features + univariate forecast + future weather
# ══════════════════════════════════════════════════════════════════════

def merge_for_multivariate(
    feature_df: pd.DataFrame,
    univariate_csv: str,
    weather_future_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Combines three sources into one extended dataframe per plant:

    ┌──────────────────┬────────────────────────────────────────────┐
    │ t < now          │ historical feature_df (all columns known)  │
    │ t+1 .. t+72      │ univariate forecast fills 'generation'     │
    │                  │ weather_future fills weather columns        │
    │                  │ time features computed from timestamp       │
    │                  │ endogenous features derived from univ fc   │
    └──────────────────┴────────────────────────────────────────────┘

    Parameters
    ----------
    feature_df        : output of build_features() — historical rows
    univariate_csv    : path to univariate_forecasts.csv
                        columns: plant_id, timestamp, forecast_mw,
                                 lower_90, upper_90, model
    weather_future_df : future weather dataframe (optional)
                        columns: timestamp, region/plant_id,
                                 irradiance_wm2, cloud_cover_pct,
                                 wind_speed_kmh, wind_direction_deg
                        If None → carries forward last known weather values

    Returns
    -------
    pd.DataFrame : extended dataframe with is_forecast flag column
    """
    feature_df = feature_df.copy()
    feature_df["timestamp"] = pd.to_datetime(feature_df["timestamp"])
    feature_df["is_forecast"] = False

    # Load univariate forecasts
    univ_df = pd.read_csv(univariate_csv, parse_dates=["timestamp"])
    print(f"[Merge] Loaded univariate forecasts: {len(univ_df):,} rows "
          f"| {univ_df['plant_id'].nunique()} plants")

    extended_parts = []

    for plant_id, hist in feature_df.groupby("plant_id"):

        hist = hist.sort_values("timestamp").reset_index(drop=True)

        # Get univariate forecast for this plant
        plant_univ = univ_df[univ_df["plant_id"] == plant_id].sort_values("timestamp")
        if len(plant_univ) == 0:
            print(f"  [WARN] No univariate forecast for {plant_id} — skipping future rows")
            extended_parts.append(hist)
            continue

        # ── Build future feature rows ─────────────────────────────
        last_row = hist.iloc[-1]
        future_rows = []

        for _, fc_row in plant_univ.iterrows():
            ts  = fc_row["timestamp"]
            row = {}

            # Timestamp-derived features
            row["timestamp"]   = ts
            row["plant_id"]    = plant_id
            row["is_forecast"] = True
            row["generation"]  = fc_row["forecast_mw"]   # univariate fills this
            row["forecast_lower_90"] = fc_row["lower_90"]
            row["forecast_upper_90"] = fc_row["upper_90"]

            # Time features — always computable
            row["hour"]        = ts.hour
            row["day_of_year"] = ts.dayofyear
            row["day_of_week"] = ts.dayofweek
            row["is_weekend"]  = int(ts.dayofweek >= 5)
            row["hour_sin"]    = np.sin(2 * np.pi * ts.hour / 24)
            row["hour_cos"]    = np.cos(2 * np.pi * ts.hour / 24)
            row["doy_sin"]     = np.sin(2 * np.pi * ts.dayofyear / 365)
            row["doy_cos"]     = np.cos(2 * np.pi * ts.dayofyear / 365)

            # Static plant metadata — copy from last historical row
            for col in ["capacity_mw", "plant_type_code", "plant_type_Hybrid",
                        "plant_type_Wind", "is_degraded", "is_offline"]:
                row[col] = last_row.get(col, 0)

            # Machine state — carry forward
            row["days_since_cleaning"] = last_row.get("days_since_cleaning", 0)
            row["curtailment_mw"]      = 0.0    # assume no curtailment in forecast

            # Weather — from future weather df or carry forward
            if weather_future_df is not None:
                w = _get_future_weather(weather_future_df, plant_id, ts)
            else:
                w = None

            if w is not None:
                row["irradiance_adjusted"] = max(0, w.get("irradiance_wm2", 0)
                                                  * (1 - w.get("cloud_cover_pct", 30) / 100))
                row["cloud_cover"]         = w.get("cloud_cover_pct", last_row.get("cloud_cover", 30))
                row["wind_speed"]          = w.get("wind_speed_kmh", last_row.get("wind_speed", 5))
                row["wind_direction"]      = w.get("wind_direction_deg", last_row.get("wind_direction", 180))
            else:
                # Carry forward last known weather + time-of-day adjustment
                row["irradiance_adjusted"] = _estimate_irradiance(ts, last_row)
                row["cloud_cover"]         = last_row.get("cloud_cover", 30)
                row["wind_speed"]          = last_row.get("wind_speed", 5)
                row["wind_direction"]      = last_row.get("wind_direction", 180)

            future_rows.append(row)

        future_df = pd.DataFrame(future_rows)

        # ── Derive endogenous features from extended series ───────
        # Combine history + future, compute lag/rolling on full series
        combined = pd.concat([hist, future_df], ignore_index=True).sort_values("timestamp")
        combined = _derive_endogenous_features(combined)

        extended_parts.append(combined)

    extended = pd.concat(extended_parts, ignore_index=True)
    n_hist   = (extended["is_forecast"] == False).sum()
    n_future = (extended["is_forecast"] == True).sum()
    print(f"[Merge] Extended df: {n_hist:,} historical + {n_future:,} forecast rows")
    return extended


def _get_future_weather(
    weather_df: pd.DataFrame,
    plant_id: str,
    ts: pd.Timestamp,
) -> dict | None:
    """Match a future weather row to plant + timestamp."""
    # Try plant-level match first, fall back to regional
    mask = weather_df["timestamp"] == ts
    if "plant_id" in weather_df.columns:
        plant_mask = mask & (weather_df["plant_id"] == plant_id)
        if plant_mask.any():
            return weather_df[plant_mask].iloc[0].to_dict()
    if mask.any():
        return weather_df[mask].iloc[0].to_dict()
    return None


def _estimate_irradiance(ts: pd.Timestamp, last_row: pd.Series) -> float:
    """Simple solar irradiance estimate when no future weather available."""
    if ts.hour < 6 or ts.hour > 18:
        return 0.0
    peak     = np.sin(np.pi * (ts.hour - 6) / 12)
    cloud    = last_row.get("cloud_cover", 30) / 100
    base_irr = 900 * peak * (1 - cloud * 0.7)
    return max(0.0, round(base_irr, 1))


def _derive_endogenous_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute lag/rolling features on the combined historical+forecast series.
    Forecast rows will use univariate forecast values in their lag lookbacks.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    g  = df["generation"]

    df["gen_lag_24"]       = g.shift(24)
    df["gen_lag_168"]      = g.shift(168)
    df["cuf"]              = (g / (df["capacity_mw"] + 1e-6)).clip(0, 1.2)
    df["ramp_rate"]        = g.diff()
    df["ramp_abs"]         = df["ramp_rate"].abs()
    df["gen_momentum_3"]   = g - g.shift(3)
    df["gen_momentum_6"]   = g - g.shift(6)
    df["gen_residual_24"]  = g - g.shift(1).rolling(24).mean()
    df["gen_normalized"]   = g / (g.shift(1).rolling(168).mean() + 1e-6)

    # Performance ratio — needs irradiance
    if "irradiance_adjusted" in df.columns:
        df["performance_ratio"] = (
            g / ((df["irradiance_adjusted"] + 1e-6) * df["capacity_mw"])
        ).clip(0, 1.2)
        df["expected_generation"] = (
            df["irradiance_adjusted"] * df["capacity_mw"] / 1000
        )

    rolling_max = g.shift(1).rolling(168).max()
    df["is_peak"] = (g >= 0.9 * rolling_max).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — STL DECOMPOSITION
# Understand trend + seasonality + residual before modelling
# ══════════════════════════════════════════════════════════════════════

def decompose_series(
    series: pd.Series,
    plant_id: str,
    period: int = 24,       # daily seasonality (hourly data)
    seasonal: int = 7,      # STL seasonal smoother (must be odd)
) -> dict:
    """
    STL (Seasonal-Trend decomposition using LOESS) on generation series.

    Why STL over classical decomposition:
      - Robust to outliers (LOESS is local regression)
      - Handles additive AND multiplicative patterns
      - Works well with solar zero-at-night series
      - Residual component exposes unexplained variance —
        high residual = hard to forecast, consider more features

    Returns dict with:
      trend, seasonal, residual : pd.Series
      seasonal_strength         : float 0–1  (1 = very seasonal)
      trend_strength            : float 0–1  (1 = strong trend)
      residual_std              : float      (unexplained noise)
      dominant_period           : int        (24h or 168h)
    """
    series = series.dropna()
    if len(series) < period * 2:
        raise ValueError(f"Series too short for STL: {len(series)} < {period*2}")

    stl = STL(
        series,
        period=period,
        seasonal=seasonal,
        trend=None,          # auto-select trend smoother
        robust=True,         # downweight outliers in LOESS
    )
    result = stl.fit()

    trend    = result.trend
    seasonal = result.seasonal
    residual = result.resid

    # Seasonal strength: Var(residual) / Var(residual + seasonal)
    # Source: Hyndman & Athanasopoulos, "Forecasting: Principles and Practice"
    var_resid    = residual.var()
    seasonal_str = max(0, 1 - var_resid / (var_resid + seasonal.var() + 1e-9))
    trend_str    = max(0, 1 - var_resid / (var_resid + trend.var()    + 1e-9))

    print(f"[{plant_id}] STL decomposition:")
    print(f"  Seasonal strength : {seasonal_str:.3f}  "
          f"({'strong' if seasonal_str > 0.6 else 'moderate' if seasonal_str > 0.3 else 'weak'})")
    print(f"  Trend strength    : {trend_str:.3f}  "
          f"({'strong' if trend_str > 0.6 else 'moderate' if trend_str > 0.3 else 'weak'})")
    print(f"  Residual std      : {residual.std():.3f} MW  "
          f"(unexplained noise)")

    return {
        "plant_id":          plant_id,
        "trend":             trend,
        "seasonal":          seasonal,
        "residual":          residual,
        "seasonal_strength": round(float(seasonal_str), 4),
        "trend_strength":    round(float(trend_str),    4),
        "residual_std":      round(float(residual.std()), 4),
        "residual_mean":     round(float(residual.mean()), 4),
        "stl_result":        result,
    }


def decompose_fleet(
    feature_df: pd.DataFrame,
    period: int = 24,
) -> pd.DataFrame:
    """
    Run STL decomposition for all plants. Returns summary DataFrame.
    Useful for understanding which plants are hard to forecast BEFORE
    spending time training models.
    """
    records = []
    for plant_id, plant_df in feature_df.groupby("plant_id"):
        series = (
            plant_df.sort_values("timestamp")
            .set_index("timestamp")["generation"]
            .asfreq("h")
            .fillna(0)
        )
        try:
            d = decompose_series(series, plant_id, period=period)
            records.append({
                "plant_id":          plant_id,
                "seasonal_strength": d["seasonal_strength"],
                "trend_strength":    d["trend_strength"],
                "residual_std":      d["residual_std"],
                "forecast_difficulty": _forecast_difficulty(d),
            })
        except Exception as e:
            print(f"  [WARN] STL failed for {plant_id}: {e}")

    summary = pd.DataFrame(records).sort_values("residual_std", ascending=False)
    print(f"\n[Decompose] Fleet STL summary:")
    print(summary.to_string(index=False))
    return summary


def _forecast_difficulty(decomp: dict) -> str:
    """Heuristic: classify how hard a plant is to forecast."""
    ss  = decomp["seasonal_strength"]
    ts  = decomp["trend_strength"]
    rs  = decomp["residual_std"]
    if ss > 0.7 and rs < 5:
        return "easy"
    elif ss > 0.4 and rs < 15:
        return "medium"
    else:
        return "hard"


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    extended_df: pd.DataFrame,
    plant_id: str,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits extended dataframe into:
      - X_hist / y_hist : historical rows for training
      - X_future        : forecast rows (t+1..t+72) for prediction

    feature_cols defaults to EXOGENOUS_FEATURES + ENDOGENOUS_FEATURES.
    Pass a custom list after running your feature reduction pipeline.

    Returns: (train_df, future_df) — both with full columns for inspection
    """
    if feature_cols is None:
        feature_cols = [
            c for c in EXOGENOUS_FEATURES + ENDOGENOUS_FEATURES
            if c in extended_df.columns
        ]

    plant_df = (
        extended_df[extended_df["plant_id"] == plant_id]
        .sort_values("timestamp")
        .dropna(subset=feature_cols)      # drop rows where any feature is NaN
        .reset_index(drop=True)
    )

    train_df  = plant_df[plant_df["is_forecast"] == False].copy()
    future_df = plant_df[plant_df["is_forecast"] == True].copy()

    print(f"[{plant_id}] Feature matrix: "
          f"train={len(train_df):,} rows, future={len(future_df):,} rows, "
          f"features={len(feature_cols)}")

    return train_df, future_df, feature_cols


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — MODELS
# ══════════════════════════════════════════════════════════════════════

def _evaluate(actual: np.ndarray, predicted: np.ndarray, name: str) -> dict:
    n    = min(len(actual), len(predicted))
    act  = actual[:n]
    pred = np.maximum(0, predicted[:n])
    mae  = mean_absolute_error(act, pred)
    rmse = np.sqrt(mean_squared_error(act, pred))
    mask = act > 0.1
    mape = float(np.mean(np.abs((act[mask]-pred[mask])/(act[mask]+1e-6)))*100) if mask.sum() else np.nan
    print(f"    {name:12s} → MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%")
    return {"model": name, "MAE": round(mae,3), "RMSE": round(rmse,3), "MAPE": round(mape,2)}


# ── LightGBM ─────────────────────────────────────────────────────────

def fit_lgbm_multivariate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    feature_names: list[str],
) -> tuple[object, dict]:
    """
    LightGBM multivariate regression.
    All features used simultaneously — no recursive step needed
    because future feature values are pre-built in X_future.
    """
    if not HAS_LGBM:
        raise ImportError("lightgbm not installed")

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        num_leaves=63,
        learning_rate=0.03,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        verbosity=-1,
        n_jobs=1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        feature_name=feature_names,
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    val_pred = model.predict(X_val)
    metrics  = _evaluate(y_val, val_pred, "lightgbm")

    # Feature importance — useful for understanding model
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    print(f"    Top features: {importance.head(5).index.tolist()}")

    return model, metrics


# ── XGBoost ──────────────────────────────────────────────────────────

def fit_xgb_multivariate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> tuple[object, dict]:
    if not HAS_XGB:
        raise ImportError("xgboost not installed")

    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        n_jobs=1,
        early_stopping_rounds=100,
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


# ── Ridge Regression ─────────────────────────────────────────────────

def fit_ridge_multivariate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> tuple[object, dict]:
    """
    Ridge regression with StandardScaler.
    Fast, interpretable, good baseline for linear relationships.
    Works well when irradiance → generation is approximately linear.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])
    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    metrics  = _evaluate(y_val, val_pred, "ridge")
    return pipe, metrics


# ── SVR ──────────────────────────────────────────────────────────────

def fit_svr_multivariate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> tuple[object, dict]:
    """
    Support Vector Regression with RBF kernel.
    Good for capturing non-linear irradiance-generation curves.
    Scales poorly with n > 10k — use on subset if needed.
    """
    n_fit = min(len(X_train), 8000)   # SVR is O(n²), cap at 8k rows
    if n_fit < len(X_train):
        print(f"    SVR: capping train to {n_fit} rows (SVR is O(n²))")
        idx = np.random.choice(len(X_train), n_fit, replace=False)
        X_fit, y_fit = X_train[idx], y_train[idx]
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
# SECTION 6 — WALK-FORWARD VALIDATION
# More realistic than random split for time series
# ══════════════════════════════════════════════════════════════════════

def walk_forward_validate(
    train_df:      pd.DataFrame,
    feature_cols:  list[str],
    val_days:      int = VAL_DAYS,
    n_splits:      int = 3,
) -> dict[str, list[dict]]:
    """
    Walk-forward (expanding window) cross-validation.

    Why walk-forward, not k-fold:
      - k-fold uses future data to predict past → data leakage
      - Walk-forward mirrors actual deployment: train on past, predict next window

    n_splits=3 means 3 successive validation windows, each val_days long.
    Final reported metrics are averaged across all windows.

    Returns: {model_name: [metrics_fold_1, metrics_fold_2, ...]}
    """
    df       = train_df.sort_values("timestamp").reset_index(drop=True)
    val_size = val_days * 24    # hours
    n        = len(df)

    # Need enough data for at least one train+val cycle
    min_train = max(val_size * 2, 500)
    if n < min_train + val_size:
        raise ValueError(
            f"Insufficient data for walk-forward validation: "
            f"need {min_train + val_size} rows, got {n}"
        )

    all_scores: dict[str, list] = {}

    for fold in range(n_splits):
        # Each fold: val window moves back by val_size from end
        val_end   = n - fold * val_size
        val_start = val_end - val_size
        if val_start < min_train:
            print(f"  [Fold {fold+1}] Skipped — insufficient training rows")
            continue

        tr = df.iloc[:val_start]
        vl = df.iloc[val_start:val_end]

        X_tr = tr[feature_cols].values
        y_tr = tr[TARGET].values
        X_vl = vl[feature_cols].values
        y_vl = vl[TARGET].values

        print(f"\n  Fold {fold+1}/{n_splits}: "
              f"train={len(tr):,} rows, val={len(vl):,} rows")

        # ── LightGBM ──
        if HAS_LGBM:
            try:
                _, m = fit_lgbm_multivariate(X_tr, y_tr, X_vl, y_vl, feature_cols)
                all_scores.setdefault("lightgbm", []).append(m)
            except Exception as e:
                print(f"    LightGBM fold {fold+1} failed: {e}")

        # ── XGBoost ──
        if HAS_XGB:
            try:
                _, m = fit_xgb_multivariate(X_tr, y_tr, X_vl, y_vl)
                all_scores.setdefault("xgboost", []).append(m)
            except Exception as e:
                print(f"    XGBoost fold {fold+1} failed: {e}")

        # ── Ridge ──
        try:
            _, m = fit_ridge_multivariate(X_tr, y_tr, X_vl, y_vl)
            all_scores.setdefault("ridge", []).append(m)
        except Exception as e:
            print(f"    Ridge fold {fold+1} failed: {e}")

        # ── SVR ──
        try:
            _, m = fit_svr_multivariate(X_tr, y_tr, X_vl, y_vl)
            all_scores.setdefault("svr", []).append(m)
        except Exception as e:
            print(f"    SVR fold {fold+1} failed: {e}")

    return all_scores


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — MODEL SELECTOR
# ══════════════════════════════════════════════════════════════════════

def select_best_multivariate_model(
    extended_df:   pd.DataFrame,
    plant_id:      str,
    feature_cols:  list[str] | None = None,
    val_days:      int = VAL_DAYS,
    n_splits:      int = 3,
) -> dict:
    """
    Full pipeline for one plant:
      1. Build feature matrix (historical + future)
      2. STL decomposition (for insight)
      3. Walk-forward validation across all models
      4. Pick best model on average MAE
      5. Refit best model on all historical data
      6. Predict on future feature rows → 72h forecast

    Returns
    -------
    dict with:
      plant_id, best_model, decomposition,
      validation_scores, avg_scores,
      forecast_df (timestamp, forecast_mw, lower_90, upper_90)
    """
    print(f"\n{'═'*60}")
    print(f"  MULTIVARIATE: {plant_id}")
    print(f"{'═'*60}")

    # ── Feature matrix ────────────────────────────────────────────
    train_df, future_df, feat_cols = build_feature_matrix(
        extended_df, plant_id, feature_cols
    )
    if len(train_df) < 500:
        raise ValueError(f"Insufficient training rows: {len(train_df)}")
    if len(future_df) == 0:
        raise ValueError(f"No future rows found — check univariate forecast was merged")

    # ── STL decomposition ─────────────────────────────────────────
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

    # ── Walk-forward validation ───────────────────────────────────
    print(f"\n  Walk-forward validation ({n_splits} folds × {val_days} days):")
    fold_scores = walk_forward_validate(train_df, feat_cols, val_days, n_splits)

    if not fold_scores:
        raise RuntimeError(f"All models failed validation for {plant_id}")

    # Average MAE across folds per model
    avg_scores = {
        model: {
            "MAE":  round(np.mean([f["MAE"]  for f in folds]), 3),
            "RMSE": round(np.mean([f["RMSE"] for f in folds]), 3),
            "MAPE": round(np.mean([f["MAPE"] for f in folds if not np.isnan(f["MAPE"])]), 2),
        }
        for model, folds in fold_scores.items()
    }

    best_name = min(avg_scores, key=lambda m: avg_scores[m]["MAE"])

    print(f"\n  Validation results (avg across {n_splits} folds):")
    for m, sc in sorted(avg_scores.items(), key=lambda x: x[1]["MAE"]):
        flag = " ← WINNER" if m == best_name else ""
        print(f"    {m:12s}: MAE={sc['MAE']:.3f}  "
              f"RMSE={sc['RMSE']:.3f}  MAPE={sc['MAPE']:.1f}%{flag}")

    # ── Refit best model on ALL training data ─────────────────────
    print(f"\n  Refitting {best_name} on full training set ({len(train_df):,} rows)...")

    X_all = train_df[feat_cols].values
    y_all = train_df[TARGET].values

    # Use last 14 days as internal val for early stopping
    split    = max(len(X_all) - val_days * 24, int(len(X_all) * 0.85))
    X_tr, X_vl = X_all[:split], X_all[split:]
    y_tr, y_vl = y_all[:split], y_all[split:]

    if best_name == "lightgbm":
        final_model, _ = fit_lgbm_multivariate(X_tr, y_tr, X_vl, y_vl, feat_cols)
    elif best_name == "xgboost":
        final_model, _ = fit_xgb_multivariate(X_tr, y_tr, X_vl, y_vl)
    elif best_name == "ridge":
        final_model, _ = fit_ridge_multivariate(X_tr, y_tr, X_vl, y_vl)
    elif best_name == "svr":
        final_model, _ = fit_svr_multivariate(X_tr, y_tr, X_vl, y_vl)

    # ── Predict on future rows ────────────────────────────────────
    X_future  = future_df[feat_cols].values
    fc_mw     = np.maximum(0, final_model.predict(X_future))

    # Confidence interval — ±1.5 × val RMSE as simple approximation
    val_rmse  = avg_scores[best_name]["RMSE"]
    fc_lower  = np.maximum(0, fc_mw - 1.5 * val_rmse)
    fc_upper  = fc_mw + 1.5 * val_rmse

    # If univariate CI tighter, use that (best of both)
    if "forecast_lower_90" in future_df.columns:
        univ_lower = future_df["forecast_lower_90"].values
        univ_upper = future_df["forecast_upper_90"].values
        fc_lower   = np.maximum(fc_lower, univ_lower)
        fc_upper   = np.minimum(fc_upper, univ_upper)
        # Where multivariate prediction is within univariate CI, trust it more
        fc_lower = np.minimum(fc_lower, fc_mw * 0.85)
        fc_upper = np.maximum(fc_upper, fc_mw * 1.15)

    forecast_df = pd.DataFrame({
        "plant_id":    plant_id,
        "timestamp":   future_df["timestamp"].values,
        "forecast_mw": np.round(fc_mw,    3),
        "lower_90":    np.round(fc_lower, 3),
        "upper_90":    np.round(fc_upper, 3),
        "model":       best_name,
    })

    print(f"\n  Forecast: {len(forecast_df)} hours | "
          f"range [{fc_mw.min():.1f}, {fc_mw.max():.1f}] MW")

    return {
        "plant_id":          plant_id,
        "best_model":        best_name,
        "feature_cols":      feat_cols,
        "decomposition":     decomp,
        "validation_scores": fold_scores,
        "avg_scores":        avg_scores,
        "final_model":       final_model,
        "forecast_df":       forecast_df,
    }


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — MONTE CARLO SCENARIO SIMULATOR
# ══════════════════════════════════════════════════════════════════════

def simulate_scenarios(
    forecast_df:   pd.DataFrame,
    decomp:        dict | None,
    n_simulations: int = 1000,
    seed:          int = 42,
) -> pd.DataFrame:
    """
    Monte Carlo simulation to generate scenario fan around point forecast.

    Three named scenarios extracted:
      P10 (worst case)  : 10th percentile of simulations
      P50 (base case)   : 50th percentile (median)
      P90 (best case)   : 90th percentile

    Uncertainty sources modelled:
      1. Weather uncertainty    : irradiance ± 15% (cloud forecast error)
      2. Model uncertainty      : ± val_rmse (from CI in forecast_df)
      3. Machine degradation    : small negative bias if plant is degraded
      4. Residual noise         : N(0, residual_std) from STL decomposition

    Parameters
    ----------
    forecast_df   : output of select_best_multivariate_model()
                    must have: timestamp, forecast_mw, lower_90, upper_90
    decomp        : STL decomposition dict (used for residual_std)
                    pass None to use ±10% as fallback noise
    n_simulations : number of Monte Carlo paths (1000 is fast and stable)
    seed          : for reproducibility

    Returns
    -------
    pd.DataFrame with columns:
      timestamp, p10, p25, p50, p75, p90,
      scenario_best, scenario_base, scenario_worst
    """
    rng       = np.random.RandomState(seed)
    n_steps   = len(forecast_df)
    fc        = forecast_df["forecast_mw"].values
    ci_width  = (forecast_df["upper_90"] - forecast_df["lower_90"]).values
    # Approximate 1 sigma from 90% CI: CI_90 ≈ ±1.645σ
    sigma_model = ci_width / (2 * 1.645)

    # Residual noise from STL — captures unexplained variance
    if decomp and "residual_std" in decomp:
        sigma_resid = decomp["residual_std"]
    else:
        sigma_resid = fc.mean() * 0.10   # 10% fallback

    # Total uncertainty per step (in quadrature)
    sigma_weather = fc * 0.08            # 8% weather forecast uncertainty
    sigma_total   = np.sqrt(
        sigma_model ** 2 +
        sigma_resid ** 2 +
        sigma_weather ** 2
    )

    # Simulate n_simulations paths
    # Shape: (n_simulations, n_steps)
    noise      = rng.randn(n_simulations, n_steps)
    # Add autocorrelation — consecutive hours are correlated
    for t in range(1, n_steps):
        noise[:, t] = 0.7 * noise[:, t-1] + 0.3 * noise[:, t]

    simulations = fc[None, :] + sigma_total[None, :] * noise
    simulations = np.maximum(0, simulations)    # generation >= 0

    # Capacity cap — can't exceed capacity (not in forecast_df, use max as proxy)
    cap_proxy   = fc.max() / 0.5               # rough capacity estimate
    simulations = np.minimum(simulations, cap_proxy)

    # Extract percentiles
    p10 = np.percentile(simulations, 10, axis=0)
    p25 = np.percentile(simulations, 25, axis=0)
    p50 = np.percentile(simulations, 50, axis=0)
    p75 = np.percentile(simulations, 75, axis=0)
    p90 = np.percentile(simulations, 90, axis=0)

    # Named scenarios
    scenario_worst = p10
    scenario_base  = p50
    scenario_best  = p90

    print(f"[Simulate] {n_simulations} Monte Carlo paths | {n_steps} steps")
    print(f"  Base (P50) total: {p50.sum():.1f} MWh")
    print(f"  Best (P90) total: {p90.sum():.1f} MWh  "
          f"(+{(p90.sum()-p50.sum())/p50.sum()*100:.1f}%)")
    print(f"  Worst(P10) total: {p10.sum():.1f} MWh  "
          f"({(p10.sum()-p50.sum())/p50.sum()*100:.1f}%)")

    result = pd.DataFrame({
        "plant_id":       forecast_df["plant_id"].values,
        "timestamp":      forecast_df["timestamp"].values,
        "forecast_mw":    fc.round(3),
        "p10":            p10.round(3),
        "p25":            p25.round(3),
        "p50":            p50.round(3),
        "p75":            p75.round(3),
        "p90":            p90.round(3),
        "scenario_worst": scenario_worst.round(3),
        "scenario_base":  scenario_base.round(3),
        "scenario_best":  scenario_best.round(3),
        "sigma_total":    sigma_total.round(3),
    })

    return result


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
    Returns structured result dict (never raises — errors are captured).
    """
    warnings.filterwarnings("ignore")
    try:
        result = select_best_multivariate_model(
            extended_df, plant_id, feature_cols, val_days, n_splits
        )
        sim_df = simulate_scenarios(
            result["forecast_df"],
            result["decomposition"],
            n_simulations=n_simulations,
        )
        decomp_summary = None
        if result["decomposition"]:
            d = result["decomposition"]
            decomp_summary = {
                "seasonal_strength": d["seasonal_strength"],
                "trend_strength":    d["trend_strength"],
                "residual_std":      d["residual_std"],
                "forecast_difficulty": _forecast_difficulty(d),
            }
        return {
            "status":           "ok",
            "plant_id":         plant_id,
            "forecast_df":      result["forecast_df"],
            "simulation_df":    sim_df,
            "best_model":       result["best_model"],
            "avg_scores":       result["avg_scores"],
            "decomp_summary":   decomp_summary,
            "error":            None,
        }
    except Exception as e:
        return {
            "status":        "error",
            "plant_id":      plant_id,
            "forecast_df":   None,
            "simulation_df": None,
            "best_model":    None,
            "avg_scores":    None,
            "decomp_summary": None,
            "error":         f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


def run_multivariate_fleet(
    feature_df:       pd.DataFrame,
    univariate_csv:   str,
    weather_future_df: pd.DataFrame | None = None,
    feature_cols:     list[str] | None = None,
    val_days:         int  = VAL_DAYS,
    n_splits:         int  = 3,
    n_simulations:    int  = 1000,
    n_jobs:           int  = -1,
    output_dir:       str  = "data/multivariate",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full multivariate forecasting pipeline for all 50 plants.

    Parameters
    ----------
    feature_df        : preprocessed + feature-engineered dataframe
    univariate_csv    : path to univariate_forecasts.csv
    weather_future_df : future weather (None = carry forward)
    feature_cols      : custom feature list (None = auto-select)
    val_days          : validation window per fold (default 14 days)
    n_splits          : walk-forward folds (default 3)
    n_simulations     : Monte Carlo paths (default 1000)
    n_jobs            : parallel workers (-1 = all cores)
    output_dir        : where to save CSVs

    Returns
    -------
    (forecast_df, simulation_df) — combined across all plants
    """
    OUT_DIR_MV = Path(output_dir)
    OUT_DIR_MV.mkdir(parents=True, exist_ok=True)

    # ── Merge once — shared across all plants ─────────────────────
    print("Merging historical features + univariate forecasts + future weather...")
    extended_df = merge_for_multivariate(feature_df, univariate_csv, weather_future_df)

    plant_ids = extended_df["plant_id"].unique().tolist()
    n_plants  = len(plant_ids)

    max_cores     = multiprocessing.cpu_count()
    resolved_jobs = max_cores if n_jobs == -1 else min(n_jobs, max_cores)
    resolved_jobs = min(resolved_jobs, n_plants)

    print(f"\n{'═'*60}")
    print(f"  MULTIVARIATE FLEET")
    print(f"  Plants      : {n_plants}")
    print(f"  Val days    : {val_days} × {n_splits} folds")
    print(f"  Simulations : {n_simulations} MC paths")
    print(f"  Workers     : {resolved_jobs} / {max_cores} cores")
    print(f"{'═'*60}\n")

    # Pass only the slice per plant into each worker
    # Pre-split to avoid sending the full 50-plant df to each process
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

    # ── Collect ───────────────────────────────────────────────────
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
            print(f"  ✓ {pid} → {res['best_model']} "
                  f"MAE={res['avg_scores'].get(res['best_model'],{}).get('MAE','?')}")
        else:
            print(f"  ✗ {pid} → {res['error'].splitlines()[0]}")
            failed.append({"plant_id": pid, "error": res["error"]})

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  FLEET COMPLETE")
    print(f"  Successful : {len(all_forecasts)} / {n_plants}")
    print(f"  Failed     : {len(failed)}")
    log_df = pd.DataFrame(log_rows)
    if len(log_df):
        print(f"\n  Model distribution:")
        for m, cnt in log_df["best_model"].value_counts().items():
            print(f"    {m:12s}: {cnt} plants ({cnt/len(log_df):.0%})")
    print(f"{'═'*60}\n")

    # ── Save ──────────────────────────────────────────────────────
    log_df.to_csv(OUT_DIR_MV / "model_selection_log.csv", index=False)
    if failed:
        pd.DataFrame(failed).to_csv(OUT_DIR_MV / "failed_plants.csv", index=False)

    if not all_forecasts:
        print("[FATAL] No multivariate forecasts generated.")
        return pd.DataFrame(), pd.DataFrame()

    forecast_df   = pd.concat(all_forecasts,   ignore_index=True)
    simulation_df = pd.concat(all_simulations, ignore_index=True)

    forecast_df.to_csv(  OUT_DIR_MV / "multivariate_forecasts.csv",  index=False)
    simulation_df.to_csv(OUT_DIR_MV / "scenario_simulations.csv",    index=False)

    print(f"  Saved:")
    print(f"    {OUT_DIR_MV}/multivariate_forecasts.csv")
    print(f"    {OUT_DIR_MV}/scenario_simulations.csv")
    print(f"    {OUT_DIR_MV}/model_selection_log.csv")

    return forecast_df, simulation_df
