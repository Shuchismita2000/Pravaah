"""
multivariate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Karnataka Renewable Energy Grid — Multivariate Forecasting Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACTUAL COLUMN SCHEMA
─────────────────────────────────────────────────────────────────────
Both historical_df_solar and forecast_df_solar share the same schema:

  Index(['plant_id', 'capacity_mw', 'timestamp', 'actual_generation_mw',
         'curtailment_mw', 'availability_mw', 'temperature', 'cloud_cover',
         'wind_speed', 'wind_direction', 'irradiance', 'health_factor',
         'plant_type_code', 'plant_type_Hybrid', 'plant_type_Solar',
         'plant_type_Wind', 'generation', 'capacity_factor',
         'generation_shortfall_mw', 'net_availability_mw',
         'health_adjusted_capacity_mw', 'is_degraded', 'is_offline',
         'generation_norm', 'hour', 'day_of_year', 'month', 'day_of_week',
         'is_weekend', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
         'month_sin', 'month_cos', 'clear_sky_irradiance', 'irradiance_adjusted',
         'irradiance_ratio', 'temp_effect', 'expected_generation',
         'performance_ratio', 'pr_rolling_7', 'pr_rolling_30',
         'days_since_cleaning', 'soiling_loss', 'adjusted_generation_signal',
         'is_daylight', 'gen_lag_1', 'gen_lag_24', 'gen_lag_168',
         'gen_rolling_mean_6', 'gen_rolling_std_6', 'gen_rolling_mean_24',
         'gen_rolling_mean_168', 'cuf', 'ramp_rate', 'ramp_abs',
         'gen_momentum_3', 'gen_momentum_6', 'daily_generation', 'load_factor',
         'is_peak', 'gen_variability_24', 'gen_variability_168', 'is_zero_gen',
         'zero_streak', 'gen_residual_24', 'gen_normalized'])

TARGET:   generation  (MW)
HORIZON:  72 hours    (t+1 .. t+72)

KEY DESIGN DECISIONS
─────────────────────────────────────────────────────────────────────
• forecast_df_solar.generation  = univariate forecast_mw proxy
  (filled upstream by run_univariate_fleet).
  forecast_df_solar does NOT have forecast_mw / lower_90 / upper_90.
• historical_df_solar.generation = actual measured generation.
• Both DataFrames are concatenated per plant with an is_forecast flag.
• Confidence intervals are produced HERE from walk-forward RMSE —
  they do not come from the input DataFrames.

SECTIONS
─────────────────────────────────────────────────────────────────────
  1.  Imports & constants
  2.  Data merger
  3.  STL decomposition
  4.  Feature matrix builder
  5.  Models: LightGBM, XGBoost, Ridge, SVR
  6.  Walk-forward validator
  7.  Model selector (per plant)
  8.  Monte Carlo scenario simulator
  9.  Fleet runner (parallelised)
  10. Public API & usage examples
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

TARGET   = "generation"   # internal name used throughout this file
HORIZON  = 72             # hours ahead
VAL_DAYS = 14             # default walk-forward validation window

# Accepted names for the target column (either is fine on input)
_TARGET_ALIASES = {"generation", "actual_generation_mw"}

# Minimum required columns — both dfs share the same schema
REQUIRED_COLS = {"plant_id", "timestamp", "generation"}

# ── Exogenous features ─────────────────────────────────────────────
# Present in forecast_df_solar — known or safely estimated for t+1..t+72
EXOGENOUS_FEATURES = [
    # Weather
    "irradiance_adjusted", "irradiance_ratio", "clear_sky_irradiance",
    "temp_effect", "cloud_cover", "wind_speed", "wind_direction", "temperature",
    # Calendar (always exact in future)
    "hour", "day_of_year", "month", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
    # Plant type — static flags
    "plant_type_code", "plant_type_Hybrid", "plant_type_Solar", "plant_type_Wind",
    # Plant health & capacity
    "capacity_mw", "is_degraded", "is_offline",
    "health_factor", "health_adjusted_capacity_mw",
    # Operational state
    "curtailment_mw", "availability_mw", "net_availability_mw",
    # Maintenance
    "days_since_cleaning", "soiling_loss",
    # Computed flags
    "is_daylight",
]

# ── Endogenous features ────────────────────────────────────────────
# Built from historical generation; forecast_df has them pre-computed
# using the univariate proxy as the generation source.
ENDOGENOUS_FEATURES = [
    "gen_lag_1", "gen_lag_24", "gen_lag_168",
    "gen_rolling_mean_6", "gen_rolling_std_6",
    "gen_rolling_mean_24", "gen_rolling_mean_168",
    "pr_rolling_7", "pr_rolling_30",
    "cuf", "load_factor", "capacity_factor",
    "generation_norm", "generation_shortfall_mw", "adjusted_generation_signal",
    "expected_generation", "performance_ratio",
    "ramp_rate", "ramp_abs", "gen_momentum_3", "gen_momentum_6",
    "gen_variability_24", "gen_variability_168",
    "gen_residual_24", "gen_normalized",
    "is_zero_gen", "zero_streak",
    "daily_generation", "is_peak",
]


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA MERGER
# ══════════════════════════════════════════════════════════════════════

def merge_for_multivariate(
    historical_df: pd.DataFrame,
    forecast_df:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Stack historical_df and forecast_df into one extended DataFrame.

    Both DataFrames have the same column schema.
      historical_df.generation = actual measured generation
      forecast_df.generation   = univariate proxy (filled upstream)

    Steps
    -----
    1. Validate required columns on both inputs.
    2. Tag: is_forecast=False (history) / True (future).
    3. Concatenate per plant, sorted by timestamp.
    4. Recompute boundary lags that cross history->forecast edge.
    5. Return combined DataFrame.

    Parameters
    ----------
    historical_df : historical_df_solar — actual generation history.
    forecast_df   : forecast_df_solar   — same schema, generation=proxy.

    Returns
    -------
    pd.DataFrame with all original columns + is_forecast flag.
    """
    historical_df = _rename_target(historical_df)
    _validate_cols(historical_df, "historical_df")
    _validate_cols(forecast_df,   "forecast_df")

    hist = historical_df.copy()
    hist["timestamp"]   = pd.to_datetime(hist["timestamp"])
    hist["is_forecast"] = False

    fc = forecast_df.copy()
    fc["timestamp"]   = pd.to_datetime(fc["timestamp"])
    fc["is_forecast"] = True

    hist_plants = set(hist["plant_id"].unique())
    fc_plants   = set(fc["plant_id"].unique())
    for pid in fc_plants - hist_plants:
        print(f"  [WARN] {pid}: in forecast_df but no history")
    for pid in hist_plants - fc_plants:
        print(f"  [WARN] {pid}: in historical_df but no forecast rows")

    print(f"[Merge] historical={len(hist):,}  forecast={len(fc):,}  "
          f"plants={len(hist_plants)}")

    parts = []
    for pid in sorted(hist_plants):
        h = hist[hist["plant_id"] == pid].copy()
        f = fc[fc["plant_id"] == pid].copy()
        if len(f) == 0:
            parts.append(h)
            continue
        combined = (
            pd.concat([h, f], ignore_index=True)
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        combined = _recompute_boundary_lags(combined)
        parts.append(combined)

    extended = pd.concat(parts, ignore_index=True)
    n_h = (~extended["is_forecast"]).sum()
    n_f = extended["is_forecast"].sum()
    print(f"[Merge] Extended: {n_h:,} historical + {n_f:,} forecast rows")
    return extended


def _rename_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise target column name to 'generation'.
    Accepts 'actual_generation_mw' or 'generation' on input.
    forecast_df will never have the target — that is fine.
    """
    if "actual_generation_mw" in df.columns and "generation" not in df.columns:
        df = df.rename(columns={"actual_generation_mw": "generation"})
    return df


def _validate_cols(df: pd.DataFrame, name: str) -> None:
    # forecast_df has no target — only require plant_id + timestamp for it
    has_target = "generation" in df.columns or "actual_generation_mw" in df.columns
    cols_needed = REQUIRED_COLS if has_target else {"plant_id", "timestamp"}
    missing = cols_needed - set(df.columns)
    if missing:
        raise ValueError(
            f"[Validate] {name} missing required columns: {missing}\n"
            f"  historical_df needs: {sorted(REQUIRED_COLS)}\n"
            f"  forecast_df needs: ['plant_id', 'timestamp'] (no target column)\n"
            f"  Got: {sorted(df.columns.tolist())}"
        )


def _recompute_boundary_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN lag/rolling values that arise at the history->forecast
    boundary.  Only fills NaN slots — clean historical values kept.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    g  = df["generation"]

    def _fill(col: str, values: pd.Series) -> None:
        if col not in df.columns:
            df[col] = values
        else:
            df[col] = df[col].combine_first(values)

    _fill("gen_lag_1",            g.shift(1))
    _fill("gen_lag_24",           g.shift(24))
    _fill("gen_lag_168",          g.shift(168))
    _fill("gen_rolling_mean_6",   g.shift(1).rolling(6,   min_periods=1).mean())
    _fill("gen_rolling_std_6",    g.shift(1).rolling(6,   min_periods=2).std().fillna(0))
    _fill("gen_rolling_mean_24",  g.shift(1).rolling(24,  min_periods=1).mean())
    _fill("gen_rolling_mean_168", g.shift(1).rolling(168, min_periods=1).mean())
    _fill("ramp_rate",            g.diff())
    _fill("ramp_abs",             g.diff().abs())
    _fill("gen_momentum_3",       g - g.shift(3))
    _fill("gen_momentum_6",       g - g.shift(6))
    _fill("gen_residual_24",      g - g.shift(1).rolling(24, min_periods=1).mean())
    rm168 = g.shift(1).rolling(168, min_periods=1).mean()
    _fill("gen_normalized",       g / (rm168 + 1e-6))
    return df


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — STL DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════

def decompose_series(
    series:        pd.Series,
    plant_id:      str,
    period:        int = 24,
    weekly_period: int = 168,
) -> dict:
    """
    STL (Seasonal-Trend decomposition using LOESS) on generation series.

    Runs two decompositions:
      - Daily  (period=24)  — intraday solar curve
      - Weekly (period=168) — weekly dispatch pattern

    Why STL over classical decomposition:
      - Robust to the many zeros in solar (night hours)
      - LOESS handles outliers gracefully
      - residual_std directly measures unexplained variance:
          high → hard to forecast, may need more features

    Parameters
    ----------
    series   : pd.Series, DatetimeIndex, hourly frequency.
               e.g. df.set_index("timestamp")["generation"].asfreq("h")
    plant_id : label for print output only.

    Returns
    -------
    dict: plant_id, trend, seasonal, residual (from daily STL),
          seasonal_strength, weekly_strength, trend_strength,
          residual_std, residual_mean, dominant_period,
          forecast_difficulty, stl_daily, stl_weekly
    """
    series = series.dropna()
    if len(series) < period * 2:
        raise ValueError(
            f"[{plant_id}] Too short for STL: need {period*2}, got {len(series)}"
        )

    def _fit(p: int):
        sw = max(7, (p // 3) | 1)   # odd, >= 7
        return STL(series, period=p, seasonal=sw, trend=None, robust=True).fit()

    stl_d    = _fit(period)
    trend    = stl_d.trend
    seasonal = stl_d.seasonal
    residual = stl_d.resid
    var_r    = residual.var()
    ss_d     = max(0.0, 1 - var_r / (var_r + seasonal.var() + 1e-9))
    ts_d     = max(0.0, 1 - var_r / (var_r + trend.var()   + 1e-9))

    stl_w = None
    ss_w  = 0.0
    if len(series) >= weekly_period * 2:
        stl_w = _fit(weekly_period)
        rw    = stl_w.resid
        ss_w  = max(0.0, 1 - rw.var() / (rw.var() + stl_w.seasonal.var() + 1e-9))

    dominant = "weekly" if ss_w > ss_d else "daily"

    out = {
        "plant_id":          plant_id,
        "trend":             trend,
        "seasonal":          seasonal,
        "residual":          residual,
        "seasonal_strength": round(float(ss_d),  4),
        "weekly_strength":   round(float(ss_w),  4),
        "trend_strength":    round(float(ts_d),  4),
        "residual_std":      round(float(residual.std()),  4),
        "residual_mean":     round(float(residual.mean()), 4),
        "dominant_period":   dominant,
        "stl_daily":         stl_d,
        "stl_weekly":        stl_w,
    }
    out["forecast_difficulty"] = _forecast_difficulty(out)

    def _label(v): return "strong" if v > 0.6 else ("moderate" if v > 0.3 else "weak")
    print(f"[{plant_id}] STL:")
    print(f"  daily seasonal : {ss_d:.3f}  ({_label(ss_d)})")
    print(f"  weekly seasonal: {ss_w:.3f}  ({_label(ss_w)})")
    print(f"  trend          : {ts_d:.3f}")
    print(f"  residual std   : {residual.std():.3f} MW")
    print(f"  dominant       : {dominant}  |  difficulty: {out['forecast_difficulty']}")
    return out


def decompose_fleet(
    historical_df: pd.DataFrame,
    output_csv:    str = "data/multivariate/stl_fleet_summary.csv",
) -> pd.DataFrame:
    """
    Run STL for all plants; return fleet-wide summary DataFrame.

    Call BEFORE training to understand plant difficulty distribution.

    Returns
    -------
    pd.DataFrame (sorted by residual_std descending):
        plant_id, seasonal_strength, weekly_strength, trend_strength,
        residual_std, dominant_period, forecast_difficulty
    """
    records = []
    for pid, pdf in historical_df.groupby("plant_id"):
        series = (
            pdf.sort_values("timestamp")
            .set_index("timestamp")[TARGET]
            .asfreq("h").fillna(0)
        )
        try:
            d = decompose_series(series, str(pid))
            records.append({k: d[k] for k in [
                "plant_id", "seasonal_strength", "weekly_strength",
                "trend_strength", "residual_std", "dominant_period",
                "forecast_difficulty",
            ]})
        except Exception as e:
            print(f"  [WARN] STL failed for {pid}: {e}")

    summary = (
        pd.DataFrame(records)
        .sort_values("residual_std", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n[Decompose] Fleet STL — {len(summary)} plants")
    for lvl in ["easy", "medium", "hard"]:
        n = (summary["forecast_difficulty"] == lvl).sum()
        print(f"  {lvl:6s}: {n} plants ({n/max(len(summary),1)*100:.0f}%)")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    return summary


def _forecast_difficulty(d: dict) -> str:
    if d["seasonal_strength"] > 0.7 and d["residual_std"] < 5:
        return "easy"
    if d["seasonal_strength"] > 0.4 and d["residual_std"] < 15:
        return "medium"
    return "hard"


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    extended_df:  pd.DataFrame,
    plant_id:     str,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Split extended DataFrame into train (history) and future (forecast).

    feature_cols defaults to EXOGENOUS_FEATURES + ENDOGENOUS_FEATURES,
    filtered to columns that actually exist.

    Training rows  : drop rows where any feature is NaN.
    Forecast rows  : ffill → bfill → fillna(0)  (must keep all HORIZON rows).

    Returns (train_df, future_df, resolved_feature_cols)
    """
    if feature_cols is None:
        all_feats    = EXOGENOUS_FEATURES + ENDOGENOUS_FEATURES
        feature_cols = [c for c in all_feats if c in extended_df.columns]

    pdf = (
        extended_df[extended_df["plant_id"] == plant_id]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    train_df  = pdf[~pdf["is_forecast"]].copy()
    future_df = pdf[ pdf["is_forecast"]].copy()

    before   = len(train_df)
    train_df = train_df.dropna(subset=feature_cols).reset_index(drop=True)
    dropped  = before - len(train_df)
    if dropped:
        print(f"  [{plant_id}] Dropped {dropped} training rows with NaN features")

    future_df[feature_cols] = (
        future_df[feature_cols].ffill().bfill().fillna(0)
    )

    print(f"[{plant_id}] train={len(train_df):,}  "
          f"future={len(future_df):,}  features={len(feature_cols)}")
    return train_df, future_df, feature_cols


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — MODELS
# ══════════════════════════════════════════════════════════════════════

def _evaluate(actual: np.ndarray, predicted: np.ndarray, name: str) -> dict:
    n    = min(len(actual), len(predicted))
    act  = actual[:n]
    pred = np.maximum(0.0, predicted[:n])
    mae  = mean_absolute_error(act, pred)
    rmse = np.sqrt(mean_squared_error(act, pred))
    mask = act > 0.1
    mape = (
        float(np.mean(np.abs((act[mask] - pred[mask]) / (act[mask] + 1e-6))) * 100)
        if mask.sum() else np.nan
    )
    print(f"    {name:12s} → MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%")
    return {"model": name, "MAE": round(mae, 3),
            "RMSE": round(rmse, 3), "MAPE": round(mape, 2)}


def fit_lgbm_multivariate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    feature_names: list[str],
) -> tuple[object, dict]:
    """LightGBM with early stopping. Prints top-5 feature importances."""
    if not HAS_LGBM:
        raise ImportError("pip install lightgbm")
    model = lgb.LGBMRegressor(
        n_estimators=1500, num_leaves=63, learning_rate=0.025,
        feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=30, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=-1, n_jobs=1,
    )
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)],
        feature_name=feature_names,
        callbacks=[lgb.early_stopping(150, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    metrics = _evaluate(y_val, model.predict(X_val), "lightgbm")
    top5 = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=False).head(5).index.tolist()
    )
    print(f"    Top-5: {top5}")
    return model, metrics


def fit_xgb_multivariate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
) -> tuple[object, dict]:
    """XGBoost with early stopping. Stronger regularisation than LGBM."""
    if not HAS_XGB:
        raise ImportError("pip install xgboost")
    model = xgb.XGBRegressor(
        n_estimators=1500, max_depth=6, learning_rate=0.025,
        subsample=0.8, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=2.0,
        random_state=42, verbosity=0, n_jobs=1,
        early_stopping_rounds=150, eval_metric="mae",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, _evaluate(y_val, model.predict(X_val), "xgboost")


def fit_ridge_multivariate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
) -> tuple[object, dict]:
    """Ridge regression + StandardScaler. Fast; wins on near-linear plants."""
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X_train, y_train)
    return pipe, _evaluate(y_val, pipe.predict(X_val), "ridge")


def fit_svr_multivariate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
) -> tuple[object, dict]:
    """SVR (RBF). Captures non-linear curves. Capped at 8k rows (O(n²))."""
    n_cap = min(len(X_train), 8_000)
    if n_cap < len(X_train):
        print(f"    SVR: sampling {n_cap}/{len(X_train)} rows")
        idx = np.random.choice(len(X_train), n_cap, replace=False)
        Xf, yf = X_train[idx], y_train[idx]
    else:
        Xf, yf = X_train, y_train
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svr", SVR(kernel="rbf", C=10, epsilon=0.5, gamma="scale"))])
    pipe.fit(Xf, yf)
    return pipe, _evaluate(y_val, pipe.predict(X_val), "svr")


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
    Walk-forward (expanding-window) cross-validation.

    Avoids data leakage of k-fold by always training on the past
    and evaluating on the next unseen window.

    Fold layout (val_size = val_days * 24 hours):
      Fold 1: train [0 .. N-3V],  val [N-3V .. N-2V]
      Fold 2: train [0 .. N-2V],  val [N-2V .. N-V ]
      Fold 3: train [0 .. N-V ],  val [N-V  .. N   ]

    Returns {model_name: [metrics_fold_1, ...]}
    """
    df        = train_df.sort_values("timestamp").reset_index(drop=True)
    val_size  = val_days * 24
    n         = len(df)
    min_train = max(val_size * 2, 500)

    if n < min_train + val_size:
        raise ValueError(
            f"Not enough data: need {min_train + val_size} rows, got {n}"
        )

    scores: dict[str, list] = {}

    for fold in range(n_splits):
        val_end   = n - fold * val_size
        val_start = val_end - val_size
        if val_start < min_train:
            print(f"  [Fold {fold+1}] Skipped — not enough training rows")
            continue

        tr, vl     = df.iloc[:val_start], df.iloc[val_start:val_end]
        X_tr, y_tr = tr[feature_cols].values, tr[TARGET].values
        X_vl, y_vl = vl[feature_cols].values, vl[TARGET].values

        print(f"\n  Fold {fold+1}/{n_splits}: train={len(tr):,}  val={len(vl):,}")

        candidates = [
            ("lightgbm", lambda: fit_lgbm_multivariate(X_tr, y_tr, X_vl, y_vl, feature_cols)
             if HAS_LGBM else None),
            ("xgboost",  lambda: fit_xgb_multivariate( X_tr, y_tr, X_vl, y_vl)
             if HAS_XGB else None),
            ("ridge",    lambda: fit_ridge_multivariate(X_tr, y_tr, X_vl, y_vl)),
            ("svr",      lambda: fit_svr_multivariate(  X_tr, y_tr, X_vl, y_vl)),
        ]
        for name, fn in candidates:
            try:
                out = fn()
                if out is None:
                    continue
                _, m = out
                scores.setdefault(name, []).append(m)
            except Exception as e:
                print(f"    {name} fold {fold+1} failed: {e}")

    return scores


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — MODEL SELECTOR
# ══════════════════════════════════════════════════════════════════════

def select_best_multivariate_model(
    extended_df:  pd.DataFrame,
    plant_id:     str,
    feature_cols: list[str] | None = None,
    val_days:     int = VAL_DAYS,
    n_splits:     int = 3,
) -> dict:
    """
    End-to-end pipeline for a single plant:

      1. Build feature matrix (train / future split)
      2. STL decomposition (diagnostic)
      3. Walk-forward CV — LightGBM, XGBoost, Ridge, SVR
      4. Select best model by average MAE
      5. Refit best model on all training data
      6. Predict generation for t+1..t+72
      7. Build 90% confidence interval from validation RMSE

    Parameters
    ----------
    extended_df  : Output of merge_for_multivariate().
    plant_id     : Single plant to process.
    feature_cols : Custom feature list (optional).
    val_days     : Days per fold (default 14).
    n_splits     : Walk-forward folds (default 3).

    Returns
    -------
    dict:
      plant_id, best_model, feature_cols,
      decomposition, validation_scores, avg_scores,
      final_model,
      forecast_df  → columns: plant_id, timestamp,
                               forecast_mw, lower_90, upper_90, model
    """
    print(f"\n{'═'*60}")
    print(f"  MULTIVARIATE: {plant_id}")
    print(f"{'═'*60}")

    train_df, future_df, feat_cols = build_feature_matrix(
        extended_df, plant_id, feature_cols
    )
    if len(train_df) < 500:
        raise ValueError(f"Too few training rows: {len(train_df)} (need ≥500)")
    if len(future_df) == 0:
        raise ValueError(
            f"No forecast rows for {plant_id}. "
            "Check forecast_df_solar has rows for this plant."
        )

    # STL decomposition (diagnostic only — does not affect model)
    series = (
        train_df.sort_values("timestamp")
        .set_index("timestamp")[TARGET]
        .asfreq("h").fillna(0)
    )
    try:
        decomp = decompose_series(series, plant_id)
    except Exception as e:
        print(f"  [WARN] STL failed: {e}")
        decomp = None

    # Walk-forward cross-validation
    print(f"\n  Walk-forward CV ({n_splits} folds × {val_days} days):")
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

    print(f"\n  Results (avg {n_splits} folds):")
    for m, sc in sorted(avg_scores.items(), key=lambda x: x[1]["MAE"]):
        flag = " ← WINNER" if m == best_name else ""
        print(f"    {m:12s}: MAE={sc['MAE']:.3f}  "
              f"RMSE={sc['RMSE']:.3f}  MAPE={sc['MAPE']:.1f}%{flag}")

    # Refit on all training data
    print(f"\n  Refitting {best_name} on {len(train_df):,} rows...")
    X_all, y_all = train_df[feat_cols].values, train_df[TARGET].values
    split        = max(len(X_all) - val_days * 24, int(len(X_all) * 0.85))
    X_tr, X_vl  = X_all[:split], X_all[split:]
    y_tr, y_vl  = y_all[:split], y_all[split:]

    fitters = {
        "lightgbm": lambda: fit_lgbm_multivariate(X_tr, y_tr, X_vl, y_vl, feat_cols),
        "xgboost":  lambda: fit_xgb_multivariate( X_tr, y_tr, X_vl, y_vl),
        "ridge":    lambda: fit_ridge_multivariate(X_tr, y_tr, X_vl, y_vl),
        "svr":      lambda: fit_svr_multivariate(  X_tr, y_tr, X_vl, y_vl),
    }
    final_model, _ = fitters[best_name]()

    # Predict t+1..t+72
    fc_mw    = np.maximum(0.0, final_model.predict(future_df[feat_cols].values))
    val_rmse = avg_scores[best_name]["RMSE"]
    lower_90 = np.maximum(0.0, fc_mw - 1.645 * val_rmse)
    upper_90 = fc_mw + 1.645 * val_rmse

    forecast_out = pd.DataFrame({
        "plant_id":    plant_id,
        "timestamp":   future_df["timestamp"].values,
        "forecast_mw": np.round(fc_mw,    3),
        "lower_90":    np.round(lower_90, 3),
        "upper_90":    np.round(upper_90, 3),
        "model":       best_name,
    })

    print(f"\n  Forecast: {len(forecast_out)} hours | "
          f"[{fc_mw.min():.1f}, {fc_mw.max():.1f}] MW")

    return {
        "plant_id":          plant_id,
        "best_model":        best_name,
        "feature_cols":      feat_cols,
        "decomposition":     decomp,
        "validation_scores": fold_scores,
        "avg_scores":        avg_scores,
        "final_model":       final_model,
        "forecast_df":       forecast_out,
    }


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — MONTE CARLO SCENARIO SIMULATOR
# ══════════════════════════════════════════════════════════════════════

def simulate_scenarios(
    forecast_df:   pd.DataFrame,
    decomp:        dict | None,
    capacity_mw:   float | None = None,
    n_simulations: int  = 1000,
    seed:          int  = 42,
) -> pd.DataFrame:
    """
    Monte Carlo simulation producing a probability fan around the
    72-hour point forecast.

    Named scenarios extracted from the fan:
      scenario_worst = P10  — bad weather / high degradation
      scenario_base  = P50  — median expected outcome
      scenario_best  = P90  — favourable conditions

    Uncertainty sources (combined in quadrature):
      1. Model error    : σ from CI width (= val RMSE × 1.645)
      2. Weather error  : 8% of point forecast
      3. Residual noise : STL residual_std (unexplained variance)
                          Falls back to 10% of mean if no decomp.

    Autocorrelation ρ=0.70 (AR-1) — cloud ramps are persistent.
    Soiling bias: 1.5% mean loss applied to all paths.

    Parameters
    ----------
    forecast_df   : Output of select_best_multivariate_model().
                    Required: plant_id, timestamp,
                              forecast_mw, lower_90, upper_90.
    decomp        : From decompose_series(). Pass None → 10% fallback.
    capacity_mw   : Physical cap. If None → 2× max forecast.
    n_simulations : Monte Carlo paths (default 1000).
    seed          : Reproducibility seed.

    Returns
    -------
    pd.DataFrame: plant_id, timestamp, forecast_mw,
                  p10, p25, p50, p75, p90,
                  scenario_worst, scenario_base, scenario_best,
                  sigma_total
    """
    rng     = np.random.RandomState(seed)
    n_steps = len(forecast_df)
    fc      = forecast_df["forecast_mw"].values.astype(float)

    ci_width     = (
        forecast_df["upper_90"].values - forecast_df["lower_90"].values
    ).astype(float)
    sigma_model  = np.maximum(ci_width / (2 * 1.645), 0.0)

    sigma_resid  = (
        float(decomp["residual_std"])
        if (decomp and "residual_std" in decomp)
        else float(fc.mean()) * 0.10
    )
    sigma_weather = fc * 0.08
    sigma_total   = np.sqrt(sigma_model**2 + sigma_resid**2 + sigma_weather**2)

    noise = rng.randn(n_simulations, n_steps)
    for t in range(1, n_steps):
        noise[:, t] = 0.70 * noise[:, t - 1] + 0.30 * noise[:, t]

    sims = fc[None, :] + sigma_total[None, :] * noise - fc[None, :] * 0.015
    sims = np.maximum(0.0, sims)
    cap  = float(capacity_mw) if capacity_mw else float(fc.max()) * 2.0
    sims = np.minimum(sims, cap)

    p10, p25 = np.percentile(sims, 10, axis=0), np.percentile(sims, 25, axis=0)
    p50, p75 = np.percentile(sims, 50, axis=0), np.percentile(sims, 75, axis=0)
    p90      = np.percentile(sims, 90, axis=0)

    d = max(float(p50.sum()), 1.0)
    print(f"[Simulate] {n_simulations} paths | {n_steps} steps")
    print(f"  Point fc : {fc.sum():.1f} MWh")
    print(f"  Base P50 : {p50.sum():.1f} MWh")
    print(f"  Best P90 : {p90.sum():.1f} MWh  (+{(p90.sum()-d)/d*100:.1f}%)")
    print(f"  Worst P10: {p10.sum():.1f} MWh  ({(p10.sum()-d)/d*100:.1f}%)")

    return pd.DataFrame({
        "plant_id":       forecast_df["plant_id"].values,
        "timestamp":      forecast_df["timestamp"].values,
        "forecast_mw":    fc.round(3),
        "p10":            p10.round(3), "p25": p25.round(3),
        "p50":            p50.round(3), "p75": p75.round(3),
        "p90":            p90.round(3),
        "scenario_worst": p10.round(3),
        "scenario_base":  p50.round(3),
        "scenario_best":  p90.round(3),
        "sigma_total":    sigma_total.round(3),
    })


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — FLEET RUNNER
# ══════════════════════════════════════════════════════════════════════
import joblib
from pathlib import Path
from datetime import datetime

BASE_MODEL_DIR = Path("models/multivariate")

def _process_one_plant(
    plant_id, plant_df, feature_cols, val_days, n_splits, n_simulations
):
    warnings.filterwarnings("ignore")

    try:
        result = select_best_multivariate_model(
            plant_df, plant_id, feature_cols,
            val_days=val_days, n_splits=n_splits
        )

        # ── Extract plant_type ───────────────────────
        plant_type = "unknown"
        if "plant_type" in plant_df.columns:
            plant_type = str(
                plant_df["plant_type"].dropna().iloc[0]
            ).strip().lower().replace(" ", "_")

        # ── Save model ──────────────────────────────
        model = result.get("model")
        if model is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M")

            model_dir = BASE_MODEL_DIR / plant_type
            model_dir.mkdir(parents=True, exist_ok=True)

            versioned_path = model_dir / f"{plant_id}_model_{ts}.joblib"
            latest_path    = model_dir / f"{plant_id}_latest.joblib"

            payload = {
                "model": model,
                "plant_id": plant_id,
                "plant_type": plant_type,
                "feature_cols": feature_cols,
                "model_name": result.get("best_model"),
            }

            joblib.dump(payload, versioned_path, compress=3)
            joblib.dump(payload, latest_path, compress=3)  # 👈 latest pointer

        # ── Simulations ─────────────────────────────
        sim_df = simulate_scenarios(
            result["forecast_df"],
            result["decomposition"],
            capacity_mw=result.get("capacity_mw"),
            n_simulations=n_simulations,
        )

        return {
            "status": "ok",
            "plant_id": plant_id,
            "forecast_df": result["forecast_df"],
            "simulation_df": sim_df,
            "best_model": result.get("best_model"),
            "avg_scores": result.get("avg_scores"),
            "decomp_summary": result.get("decomp_summary"),
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "plant_id": plant_id,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


def run_multivariate_fleet(
    historical_df: pd.DataFrame,
    forecast_df:   pd.DataFrame,
    feature_cols:  list[str] | None = None,
    val_days:      int  = VAL_DAYS,
    n_splits:      int  = 3,
    n_simulations: int  = 1000,
    n_jobs:        int  = -1,
    output_dir:    str  = "data/multivariate",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full multivariate forecasting pipeline for all plants.

    Parameters
    ----------
    historical_df : historical_df_solar
                    Same schema shown at the top of this file.
                    generation = actual measured generation.

    forecast_df   : forecast_df_solar
                    Same schema as historical_df_solar.
                    generation = univariate proxy (filled upstream).
                    *** Does NOT need forecast_mw / lower_90 / upper_90 ***
                    Those columns are produced by this module, not consumed.

    feature_cols  : Custom feature list (optional).
                    Default: EXOGENOUS_FEATURES + ENDOGENOUS_FEATURES
                             filtered to existing columns.

    val_days      : Days per walk-forward fold (default 14).
    n_splits      : Walk-forward folds (default 3).
    n_simulations : Monte Carlo paths per plant (default 1000).
    n_jobs        : Parallel workers (-1 = all cores).
    output_dir    : Where to write output CSVs.

    Returns
    -------
    (all_forecasts_df, all_simulations_df)

    all_forecasts_df columns:
        plant_id, timestamp, forecast_mw, lower_90, upper_90, model

    all_simulations_df columns:
        plant_id, timestamp, forecast_mw,
        p10, p25, p50, p75, p90,
        scenario_worst, scenario_base, scenario_best, sigma_total

    Saved files:
        {output_dir}/multivariate_forecasts.csv
        {output_dir}/scenario_simulations.csv
        {output_dir}/model_selection_log.csv
        {output_dir}/failed_plants.csv  (only if failures)
    """
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    historical_df = _rename_target(historical_df)
    _validate_cols(historical_df, "historical_df")
    _validate_cols(forecast_df,   "forecast_df")

    print("\nMerging DataFrames...")
    extended = merge_for_multivariate(historical_df, forecast_df)

    plant_ids     = sorted(extended["plant_id"].unique().tolist())
    n_plants      = len(plant_ids)
    max_cores     = multiprocessing.cpu_count()
    rj            = max_cores if n_jobs == -1 else min(n_jobs, max_cores)
    rj            = min(rj, n_plants)

    print(f"\n{'═'*60}")
    print(f"  MULTIVARIATE FLEET RUN")
    print(f"  Plants      : {n_plants}")
    print(f"  Target      : {TARGET}  (MW)")
    print(f"  Horizon     : t+1 .. t+{HORIZON}h")
    print(f"  Val window  : {val_days} days × {n_splits} folds")
    print(f"  MC paths    : {n_simulations} / plant")
    print(f"  Workers     : {rj} / {max_cores} cores")
    print(f"{'═'*60}\n")

    slices = {
        pid: extended[extended["plant_id"] == pid].copy()
        for pid in plant_ids
    }

    results = Parallel(n_jobs=rj, backend="loky", verbose=5)(
        delayed(_process_one_plant)(
            pid, slices[pid], feature_cols, val_days, n_splits, n_simulations
        )
        for pid in plant_ids
    )

    fc_parts, sim_parts, log_rows, failed = [], [], [], []

    for res in results:
        pid = res["plant_id"]
        if res["status"] == "ok":
            fc_parts.append(res["forecast_df"])
            sim_parts.append(res["simulation_df"])
            row = {"plant_id": pid, "best_model": res["best_model"]}
            for m, sc in (res["avg_scores"] or {}).items():
                row[f"{m}_MAE"]  = sc["MAE"]
                row[f"{m}_MAPE"] = sc["MAPE"]
            if res["decomp_summary"]:
                row.update(res["decomp_summary"])
            log_rows.append(row)
            mae = (res["avg_scores"] or {}).get(res["best_model"], {}).get("MAE", "?")
            print(f"  ✓ {pid}  model={res['best_model']}  MAE={mae}")
        else:
            print(f"  ✗ {pid}  {res['error'].splitlines()[0]}")
            failed.append({"plant_id": pid, "error": res["error"]})

    log_df = pd.DataFrame(log_rows)
    print(f"\n{'═'*60}")
    print(f"  FLEET COMPLETE  {len(fc_parts)}/{n_plants} succeeded")
    if len(log_df):
        print("\n  Model selection:")
        for m, cnt in log_df["best_model"].value_counts().items():
            print(f"    {m:12s}: {cnt} plants ({cnt/len(log_df):.0%})")
        if "forecast_difficulty" in log_df.columns:
            print("\n  Forecast difficulty:")
            for lvl, cnt in log_df["forecast_difficulty"].value_counts().items():
                print(f"    {lvl:8s}: {cnt} plants")
    print(f"{'═'*60}\n")

    log_df.to_csv(OUT / "model_selection_log.csv", index=False)
    if failed:
        pd.DataFrame(failed).to_csv(OUT / "failed_plants.csv", index=False)

    if not fc_parts:
        print("[FATAL] No forecasts generated.")
        return pd.DataFrame(), pd.DataFrame()

    all_fc  = pd.concat(fc_parts,  ignore_index=True)
    all_sim = pd.concat(sim_parts, ignore_index=True)
    all_fc.to_csv( OUT / "multivariate_forecasts.csv",  index=False)
    all_sim.to_csv(OUT / "scenario_simulations.csv",    index=False)
    print(f"  Saved to {OUT}/")
    return all_fc, all_sim


# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — PUBLIC API
# ══════════════════════════════════════════════════════════════════════
import joblib
from pathlib import Path
from datetime import datetime

def run_single_plant(
    historical_df: pd.DataFrame,
    forecast_df:   pd.DataFrame,
    plant_id:      str,
    n_simulations: int = 1000,
    feature_cols:  list[str] | None = None,
    plant_type:    str = "unknown",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full pipeline for ONE plant. Useful for debugging.

    Parameters
    ----------
    historical_df : historical_df_solar
    forecast_df   : forecast_df_solar (same schema, generation=proxy)
    plant_id      : e.g. "KA_SOLAR_001"
    n_simulations : Monte Carlo paths
    feature_cols  : Optional custom feature list

    Returns
    -------
    (forecast_df_out, simulation_df)

    forecast_df_out: plant_id, timestamp, forecast_mw, lower_90, upper_90, model
    simulation_df  : plant_id, timestamp, forecast_mw,
                     p10, p25, p50, p75, p90,
                     scenario_worst, scenario_base, scenario_best, sigma_total

    Example
    -------
    >>> fc, sim = run_single_plant(
    ...     historical_df_solar, forecast_df_solar, "KA_SOLAR_001"
    ... )
    >>> print(fc[["timestamp", "forecast_mw", "lower_90", "upper_90"]])
    >>> print(sim[["timestamp", "p10", "p50", "p90"]].head(24))
    """
    extended = merge_for_multivariate(historical_df, forecast_df)
    result   = select_best_multivariate_model(extended, plant_id, feature_cols)

    cap_mw = None
    mask   = historical_df["plant_id"] == plant_id
    if mask.any() and "capacity_mw" in historical_df.columns:
        cap_mw = float(historical_df.loc[mask, "capacity_mw"].iloc[0])

     # ── Save model ───────────────────────────────
    model = result.get("model", None)
    if model is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")

        model_dir = Path("models/multivariate") / plant_type
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{plant_id}_model_{ts}.joblib"

        joblib.dump(
            {
                "model": model,
                "plant_id": plant_id,
                "plant_type": plant_type,
                "capacity_mw": cap_mw,
                "feature_cols": feature_cols,
                "model_name": result.get("model_name"),
            },
            model_path,
            compress=3,
        )


    sim_df = simulate_scenarios(
        result["forecast_df"], result["decomposition"],
        capacity_mw=cap_mw, n_simulations=n_simulations,
    )
    return result["forecast_df"], sim_df


def run_decomposition_only(
    historical_df: pd.DataFrame,
    output_csv:    str = "data/multivariate/stl_fleet_summary.csv",
) -> pd.DataFrame:
    """
    Run STL decomposition fleet-wide WITHOUT training any models.

    Use before training to:
      - Identify easy / medium / hard plants
      - Spot high residual-variance plants (may need extra features)
      - See whether daily or weekly seasonality dominates

    Returns
    -------
    pd.DataFrame: plant_id, seasonal_strength, weekly_strength,
                  trend_strength, residual_std,
                  dominant_period, forecast_difficulty

    Example
    -------
    >>> stl = run_decomposition_only(historical_df_solar)
    >>> hard = stl[stl["forecast_difficulty"] == "hard"]
    >>> print(hard[["plant_id", "residual_std"]].to_string())
    """
    return decompose_fleet(historical_df, output_csv=output_csv)