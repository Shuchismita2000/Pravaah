"""
curtailment.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Curtailment forecasting module

Why curtailment matters:
  If a plant can generate 50 MW but grid operator curtails 8 MW,
  actual metered output = 42 MW.
  Your model trained on actual generation already has curtailment
  embedded — but the FUTURE curtailment is unknown.
  Forecasting it separately lets you:
    1. Correct gross generation forecast → net deliverable forecast
    2. Flag high-curtailment hours as uncertain
    3. Give grid operators actionable insight on expected curtailment

Curtailment characteristics that make it forecastable:
  - Strongly time-of-day driven (peak solar = peak curtailment risk)
  - Seasonal (monsoon hydro competes with solar → higher curtailment)
  - Regional (transmission-constrained zones curtail more)
  - Plant-type specific (solar curtailed more than wind in Karnataka)
  - Weekend effect (lower demand → grid absorbs less)

Two-stage approach:
  Stage 1 — Classification: will there be any curtailment? (binary)
  Stage 2 — Regression   : how much? (MW amount, given Stage 1 = yes)

  This handles the zero-inflation problem — curtailment is 0 most hours.
  A single regression model struggles with this because it tries to
  predict near-zero for most rows and misses the spike structure.
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    f1_score, precision_score, recall_score
)
from joblib import Parallel, delayed

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

OUT_DIR = Path("data/curtailment")


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — FEATURE BUILDER
# Curtailment features are different from generation features
# ══════════════════════════════════════════════════════════════════════

def build_curtailment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features specifically predictive of curtailment events.

    Input df must have:
        timestamp, plant_id, plant_type, curtailment_mw,
        actual_generation_mw, capacity_mw,
        irradiance_adjusted (or irradiance), cloud_cover, wind_speed

    Why these features and not the full 32-feature set:
        Curtailment is a GRID decision, not a weather-physics decision.
        The most predictive features are:
          - Time of day + day of week  (grid load patterns)
          - Generation as % of capacity (how full the grid is)
          - Seasonal  (monsoon hydro competition)
          - Regional aggregate generation (not just this plant)
          - Recent curtailment history (operators repeat patterns)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["plant_id", "timestamp"]).reset_index(drop=True)

    # ── Target engineering ────────────────────────────────────────────
    # Binary: was there any curtailment this hour?
    df["curtailed"]        = (df["curtailment_mw"] > 0.1).astype(int)
    # Amount (only meaningful when curtailed == 1)
    df["curtailment_mw"]   = df["curtailment_mw"].clip(lower=0)

    # ── Time features ─────────────────────────────────────────────────
    df["hour"]         = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["month"]        = df["timestamp"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_solar"]= ((df["hour"] >= 10) & (df["hour"] <= 14)).astype(int)
    df["is_monsoon"]   = df["month"].isin([6, 7, 8, 9]).astype(int)

    # Cyclical encoding
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Generation pressure features ──────────────────────────────────
    # How hard is the plant pushing against its capacity?
    # High generation pressure → higher curtailment risk
    if "actual_generation_mw" in df.columns and "capacity_mw" in df.columns:
        df["gen_pressure"] = (
            df["actual_generation_mw"] / (df["capacity_mw"] + 1e-6)
        ).clip(0, 1.2)
    elif "generation" in df.columns:
        df["gen_pressure"] = (
            df["generation"] / (df["capacity_mw"] + 1e-6)
        ).clip(0, 1.2)

    # Irradiance as proxy for potential generation pressure
    irr_col = "irradiance_adjusted" if "irradiance_adjusted" in df.columns else "irradiance_wm2"
    if irr_col in df.columns:
        df["irradiance_norm"] = (df[irr_col] / 1000).clip(0, 1)  # normalise to 0-1
    else:
        df["irradiance_norm"] = df["is_peak_solar"] * 0.7         # fallback estimate

    # ── Lag features — curtailment has strong autocorrelation ─────────
    # If curtailed last hour, likely curtailed this hour too
    for lag in [1, 2, 3, 24, 48, 168]:
        df[f"curtailed_lag_{lag}"]   = df["curtailed"].shift(lag)
        df[f"curtail_mw_lag_{lag}"]  = df["curtailment_mw"].shift(lag)

    # Rolling curtailment rate — what % of hours in past 24h were curtailed?
    df["curtail_rate_24h"]  = df["curtailed"].shift(1).rolling(24).mean()
    df["curtail_rate_168h"] = df["curtailed"].shift(1).rolling(168).mean()

    # Rolling mean curtailment amount
    df["curtail_mean_24h"]  = df["curtailment_mw"].shift(1).rolling(24).mean()

    # ── Plant type encoding ───────────────────────────────────────────
    type_map = {"Solar": 0, "Wind": 1, "Hybrid": 2}
    df["plant_type_code"] = df["plant_type"].map(type_map).fillna(0).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df


# Features used for both stages
CURTAILMENT_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "is_peak_solar", "is_monsoon",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "gen_pressure", "irradiance_norm",
    "curtailed_lag_1", "curtailed_lag_2", "curtailed_lag_3",
    "curtailed_lag_24", "curtailed_lag_168",
    "curtail_mw_lag_1", "curtail_mw_lag_24", "curtail_mw_lag_168",
    "curtail_rate_24h", "curtail_rate_168h",
    "curtail_mean_24h",
    "plant_type_code",
]


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — TWO-STAGE MODEL
# Stage 1: will curtailment happen? (classifier)
# Stage 2: how much curtailment? (regressor, run only when Stage 1 = yes)
# ══════════════════════════════════════════════════════════════════════

def fit_curtailment_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,      # binary: curtailed or not
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> tuple:
    """
    Stage 1: Binary classifier — will this hour have curtailment?

    LightGBM if available (handles class imbalance well via scale_pos_weight),
    falls back to Gradient Boosting.

    Returns: (model, val_metrics_dict)
    """
    curtail_rate = y_train.mean()
    print(f"    Curtailment rate in train: {curtail_rate:.1%}")

    if HAS_LGBM:
        # scale_pos_weight compensates for class imbalance
        # (most hours have 0 curtailment)
        pos_weight = (1 - curtail_rate) / (curtail_rate + 1e-6)
        model = lgb.LGBMClassifier(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, random_state=42,
            )),
        ])
        model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        "f1":        round(f1_score(y_val, val_pred, zero_division=0), 3),
        "precision": round(precision_score(y_val, val_pred, zero_division=0), 3),
        "recall":    round(recall_score(y_val, val_pred, zero_division=0), 3),
        "curtail_rate_train": round(float(curtail_rate), 3),
        "curtail_rate_val":   round(float(y_val.mean()), 3),
    }
    print(f"    Classifier: F1={metrics['f1']:.3f}  "
          f"Precision={metrics['precision']:.3f}  "
          f"Recall={metrics['recall']:.3f}")

    return model, val_prob, metrics


def fit_curtailment_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,      # continuous: curtailment_mw (only curtailed hours)
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> tuple:
    """
    Stage 2: Regressor — how much curtailment (MW)?
    Trained ONLY on hours where curtailment > 0.

    Why separate from classifier:
        If you train one regression model on all hours including zeros,
        it learns to predict near-zero for most hours and underestimates
        when curtailment actually happens — exactly the wrong behavior.
    """
    if len(y_train) < 30:
        # Not enough curtailment events to fit a regressor
        # Fall back to mean of training curtailment
        mean_curt = float(y_train.mean()) if len(y_train) > 0 else 0.0
        print(f"    Regressor: too few events ({len(y_train)}) → using mean={mean_curt:.2f} MW")
        return None, mean_curt, {}

    if HAS_LGBM:
        model = lgb.LGBMRegressor(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)] if len(y_val) > 0 else None,
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)]
            if len(y_val) > 0 else [],
        )
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    GradientBoostingRegressor(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, random_state=42,
            )),
        ])
        model.fit(X_train, y_train)

    if len(y_val) > 0:
        val_pred = np.maximum(0, model.predict(X_val))
        mae  = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        metrics = {"MAE": round(mae, 3), "RMSE": round(rmse, 3)}
        print(f"    Regressor: MAE={mae:.3f}  RMSE={rmse:.3f} MW  "
              f"(trained on {len(y_train)} curtailed hours)")
    else:
        metrics = {}

    return model, None, metrics


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — PER-PLANT CURTAILMENT FORECAST
# ══════════════════════════════════════════════════════════════════════

def forecast_curtailment_one_plant(
    plant_id: str,
    plant_df: pd.DataFrame,
    future_features_df: pd.DataFrame,
    val_days: int = 14,
) -> dict:
    """
    Fits the two-stage curtailment model for one plant and produces
    a 72-hour curtailment forecast.

    Parameters
    ----------
    plant_id           : plant identifier
    plant_df           : historical feature dataframe for this plant
                         (output of build_curtailment_features)
    future_features_df : feature rows for the 72-hour forecast window
                         (time + gen_pressure from univariate forecast)
    val_days           : validation window in days

    Returns
    -------
    dict with:
        plant_id, classifier_metrics, regressor_metrics,
        forecast_df (timestamp, curtailment_mw_forecast, curtail_probability)
    """
    print(f"\n  [{plant_id}] Fitting curtailment model...")

    feat_cols = [c for c in CURTAILMENT_FEATURES if c in plant_df.columns]
    if len(feat_cols) < 5:
        raise ValueError(f"Too few curtailment features available: {feat_cols}")

    # ── Time split ────────────────────────────────────────────────────
    plant_df = plant_df.sort_values("timestamp").reset_index(drop=True)
    val_size = val_days * 24
    split    = len(plant_df) - val_size

    train_df = plant_df.iloc[:split]
    val_df   = plant_df.iloc[split:]

    X_tr = train_df[feat_cols].values
    X_vl = val_df[feat_cols].values
    y_clf_tr = train_df["curtailed"].values
    y_clf_vl = val_df["curtailed"].values
    y_reg_tr = train_df["curtailment_mw"].values
    y_reg_vl = val_df["curtailment_mw"].values

    # ── Stage 1: Classifier ───────────────────────────────────────────
    clf, val_probs, clf_metrics = fit_curtailment_classifier(
        X_tr, y_clf_tr, X_vl, y_clf_vl
    )

    # ── Stage 2: Regressor (on curtailed hours only) ──────────────────
    # Train only on hours where curtailment actually happened
    curtailed_mask_tr = y_clf_tr == 1
    curtailed_mask_vl = y_clf_vl == 1

    reg, fallback_mean, reg_metrics = fit_curtailment_regressor(
        X_tr[curtailed_mask_tr], y_reg_tr[curtailed_mask_tr],
        X_vl[curtailed_mask_vl], y_reg_vl[curtailed_mask_vl],
    )

    # ── Predict on future window ──────────────────────────────────────
    future_feat_cols = [c for c in feat_cols if c in future_features_df.columns]
    missing_feats = set(feat_cols) - set(future_feat_cols)

    if missing_feats:
        # Fill missing future features with median from training
        for col in missing_feats:
            if col in train_df.columns:
                future_features_df[col] = train_df[col].median()
            else:
                future_features_df[col] = 0

    X_future  = future_features_df[feat_cols].values

    # Stage 1: probability of curtailment per hour
    curt_prob = clf.predict_proba(X_future)[:, 1]

    # Stage 2: expected curtailment amount
    if reg is not None:
        curt_amount = np.maximum(0, reg.predict(X_future))
    else:
        curt_amount = np.full(len(X_future), fallback_mean)

    # Combined forecast: probability × amount
    # This gives expected curtailment in MW per hour
    curt_forecast = curt_prob * curt_amount

    forecast_df = pd.DataFrame({
        "plant_id":              plant_id,
        "timestamp":             future_features_df["timestamp"].values,
        "curtailment_mw_forecast": np.round(curt_forecast, 3),
        "curtail_probability":   np.round(curt_prob, 3),
        "curtail_amount_given_event": np.round(curt_amount, 3),
    })

    print(f"  [{plant_id}] Curtailment forecast: "
          f"avg={curt_forecast.mean():.2f} MW | "
          f"max={curt_forecast.max():.2f} MW | "
          f"curtailed hours={int((curt_prob > 0.5).sum())}/72")

    return {
        "plant_id":          plant_id,
        "classifier_metrics": clf_metrics,
        "regressor_metrics":  reg_metrics,
        "forecast_df":        forecast_df,
    }


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — BUILD FUTURE FEATURES FOR CURTAILMENT FORECAST
# ══════════════════════════════════════════════════════════════════════

def build_curtailment_future_features(
    hist_df: pd.DataFrame,
    plant_id: str,
    plant_fc_df: pd.DataFrame,   # ← was univariate_forecast_df, now pre-filtered
    capacity_mw: float,
    plant_type: str,
) -> pd.DataFrame:

    # Replace the old univ filter block at the top:
    # OLD:
    #   univ = univariate_forecast_df[
    #       univariate_forecast_df["plant_id"] == plant_id
    #   ].sort_values("timestamp").reset_index(drop=True)

    # NEW — already filtered, just sort:
    univ = plant_fc_df.sort_values("timestamp").reset_index(drop=True)

    if len(univ) == 0:
        raise ValueError(f"No forecast rows for {plant_id}")


    hist = hist_df.sort_values("timestamp").reset_index(drop=True)
    last_known = hist.iloc[-1]

    type_map = {"Solar": 0, "Wind": 1, "Hybrid": 2}
    rows = []

    for i, (_, fc_row) in enumerate(univ.iterrows()):
        ts = pd.Timestamp(fc_row["timestamp"])

        # Time features
        row = {
            "timestamp":     ts,
            "hour":          ts.hour,
            "day_of_week":   ts.dayofweek,
            "month":         ts.month,
            "is_weekend":    int(ts.dayofweek >= 5),
            "is_peak_solar": int(10 <= ts.hour <= 14),
            "is_monsoon":    int(ts.month in [6, 7, 8, 9]),
            "hour_sin":      np.sin(2 * np.pi * ts.hour / 24),
            "hour_cos":      np.cos(2 * np.pi * ts.hour / 24),
            "month_sin":     np.sin(2 * np.pi * ts.month / 12),
            "month_cos":     np.cos(2 * np.pi * ts.month / 12),
        }

        # Generation pressure from univariate forecast
        row["gen_pressure"] = min(1.2, fc_row["forecast_mw"] / (capacity_mw + 1e-6))

        # Irradiance proxy from hour (no future weather = use time-based estimate)
        if ts.hour >= 6 and ts.hour <= 18:
            row["irradiance_norm"] = np.sin(np.pi * (ts.hour - 6) / 12) * row["gen_pressure"]
        else:
            row["irradiance_norm"] = 0.0

        # Lag features — use historical values where available
        # For future lags (e.g., lag_1 for first future hour), carry forward last known
        hist_curtailed = hist["curtailed"].values if "curtailed" in hist.columns else np.zeros(len(hist))
        hist_curt_mw   = hist["curtailment_mw"].values

        for lag in [1, 2, 3, 24, 48, 168]:
            idx = -(lag - i)   # negative index into history
            if idx < -len(hist):
                row[f"curtailed_lag_{lag}"]  = 0.0
                row[f"curtail_mw_lag_{lag}"] = 0.0
            else:
                row[f"curtailed_lag_{lag}"]  = float(hist_curtailed[idx])
                row[f"curtail_mw_lag_{lag}"] = float(hist_curt_mw[idx])

        # Rolling rates from historical tail
        row["curtail_rate_24h"]  = float(hist_curtailed[-24:].mean()) if len(hist_curtailed) >= 24 else 0.0
        row["curtail_rate_168h"] = float(hist_curtailed[-168:].mean()) if len(hist_curtailed) >= 168 else 0.0
        row["curtail_mean_24h"]  = float(hist_curt_mw[-24:].mean()) if len(hist_curt_mw) >= 24 else 0.0
        row["plant_type_code"]   = type_map.get(plant_type, 0)

        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — APPLY CURTAILMENT CORRECTION TO GENERATION FORECAST
# ══════════════════════════════════════════════════════════════════════

def apply_curtailment_correction(
    generation_forecast_df: pd.DataFrame,
    curtailment_forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Subtracts curtailment forecast from gross generation forecast.

    generation_forecast_df  : output of univariate or multivariate model
                              columns: plant_id, timestamp, forecast_mw, ...
    curtailment_forecast_df : output of forecast_curtailment_one_plant()
                              columns: plant_id, timestamp,
                                       curtailment_mw_forecast

    Returns: corrected forecast with additional columns:
        gross_forecast_mw     — original generation forecast
        curtailment_forecast  — expected curtailment
        net_forecast_mw       — gross - curtailment (what actually gets metered)
        curtail_probability   — how likely curtailment is each hour
    """
    gen = generation_forecast_df.copy()
    cur = curtailment_forecast_df[
        ["plant_id", "timestamp", "curtailment_mw_forecast", "curtail_probability"]
    ].copy()

    gen["timestamp"] = pd.to_datetime(gen["timestamp"])
    cur["timestamp"] = pd.to_datetime(cur["timestamp"])

    merged = gen.merge(cur, on=["plant_id", "timestamp"], how="left")
    merged["curtailment_mw_forecast"] = merged["curtailment_mw_forecast"].fillna(0)
    merged["curtail_probability"]     = merged["curtail_probability"].fillna(0)

    # Rename original forecast
    merged = merged.rename(columns={"forecast_mw": "gross_forecast_mw"})

    # Net = gross - curtailment (floor at 0)
    merged["net_forecast_mw"] = np.maximum(
        0,
        merged["gross_forecast_mw"] - merged["curtailment_mw_forecast"]
    )

    # Adjust confidence intervals if present
    if "lower_90" in merged.columns:
        merged["lower_90"] = np.maximum(0, merged["lower_90"] - merged["curtailment_mw_forecast"])
    if "upper_90" in merged.columns:
        merged["upper_90"] = np.maximum(0, merged["upper_90"] - merged["curtailment_mw_forecast"])

    print(f"\n[Curtailment correction applied]")
    for pid, grp in merged.groupby("plant_id"):
        total_gross = grp["gross_forecast_mw"].sum()
        total_curt  = grp["curtailment_mw_forecast"].sum()
        total_net   = grp["net_forecast_mw"].sum()
        if total_curt > 0.1:
            print(f"  {pid}: gross={total_gross:.1f} MWh | "
                  f"curtailment={total_curt:.1f} MWh ({total_curt/total_gross*100:.1f}%) | "
                  f"net={total_net:.1f} MWh")

    return merged


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — FLEET RUNNER
# ══════════════════════════════════════════════════════════════════════
import joblib
from pathlib import Path
from datetime import datetime

MODEL_DIR = Path("models/curtailment")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

def _process_curtailment_one_plant(
    plant_id: str,
    plant_df: pd.DataFrame,
    future_df: pd.DataFrame,
    capacity_mw: float,
    plant_type: str,
    val_days: int,
) -> dict:
    """Worker for joblib parallelisation."""
    warnings.filterwarnings("ignore")
    try:
        result = forecast_curtailment_one_plant(
            plant_id, plant_df, future_df, val_days
        )
         # ── Save model ───────────────────────────────
        model = result.get("model", None)
        if model is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            model_path = MODEL_DIR / f"{plant_id}_curtailment_model_{ts}.joblib"

            joblib.dump(
                {
                    "model": model,
                    "plant_id": plant_id,
                    "plant_type": plant_type,
                    "capacity_mw": float(capacity_mw),
                    "val_days": val_days,
                },
                model_path,
                compress=3,
            )

        # ── Return success result ────────────────────
        return {
            "status": "ok",
            "plant_id": plant_id,
            "forecast_df": result.get("forecast_df"),
        }
    
    except Exception as e:
        import traceback
        return {
            "status": "error", "plant_id": plant_id,
            "forecast_df": None,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }

def run_curtailment_fleet(
    curtailment_input_df: pd.DataFrame,   # single merged dataframe — see below
    val_days: int = 14,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Run curtailment forecasting for all plants in parallel.

    Parameters
    ----------
    curtailment_input_df : Single merged dataframe with ALL required columns.
                           Prepare it BEFORE calling this function:

                           -- Plant identity --
                           plant_id              str       e.g. "PLANT_001"
                           plant_type            str       "Solar" / "Wind" / "Hybrid"
                           capacity_mw           float     nameplate capacity, constant per plant

                           -- Historical rows (past actuals, one row per plant per hour) --
                           timestamp             datetime  hourly, no gaps
                           curtailment_mw        float     actual curtailment, ≥ 0
                           actual_generation_mw  float     actual metered generation
                           irradiance_adjusted   float     W/m²  (or irradiance_wm2)

                           -- Forecast rows (future 72 hours, one row per plant per hour) --
                           timestamp             datetime  next 72 hours
                           forecast_mw           float     univariate generation forecast
                           is_forecast_row       bool/int  1 = future row, 0 = historical row

                           How to build it:
                           ───────────────
                           hist = your_historical_df.copy()
                           hist["is_forecast_row"] = 0
                           hist["forecast_mw"]     = hist["actual_generation_mw"]

                           fc = univariate_forecast_df.copy()   # output of run_univariate_fleet()
                           fc = fc.merge(
                               plant_master_df[["plant_id", "plant_type", "capacity_mw"]],
                               on="plant_id", how="left"
                           )
                           fc["is_forecast_row"]      = 1
                           fc["curtailment_mw"]       = 0.0
                           fc["actual_generation_mw"] = fc["forecast_mw"]
                           fc["irradiance_adjusted"]  = 0.0   # unknown for future

                           curtailment_input_df = pd.concat([hist, fc], ignore_index=True)

    val_days : Last N days of historical rows used for validation. Default 14.
    n_jobs   : Parallel workers. -1 = all cores.

    Returns
    -------
    pd.DataFrame : curtailment forecast for all plants with columns:
                   plant_id, timestamp, curtailment_mw_forecast,
                   curtail_probability, curtail_amount_given_event
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Validate required columns ─────────────────────────────────────
    required = {
        "plant_id", "plant_type", "capacity_mw",
        "timestamp", "curtailment_mw", "actual_generation_mw",
        "forecast_mw", "is_forecast_row",
    }
    missing = required - set(curtailment_input_df.columns)
    if missing:
        raise ValueError(
            f"curtailment_input_df missing columns: {missing}\n"
            f"See docstring for how to build this dataframe."
        )

    # ── Normalize ─────────────────────────────────────────────────────
    df = curtailment_input_df.copy()
    df["timestamp"]  = pd.to_datetime(df["timestamp"])
    df["plant_type"] = df["plant_type"].str.strip().str.title()

    # ── Split historical vs forecast rows ─────────────────────────────
    hist_df = df[df["is_forecast_row"] == 0].copy()
    fc_df   = df[df["is_forecast_row"] == 1].copy()

    if len(hist_df) == 0:
        raise ValueError("No historical rows found (is_forecast_row == 0). Check your input.")
    if len(fc_df) == 0:
        raise ValueError("No forecast rows found (is_forecast_row == 1). Check your input.")

    # ── Build curtailment features on historical rows only ────────────
    print("Building curtailment features...")
    curt_feat_df = build_curtailment_features(hist_df)

    # ── Build future feature rows per plant ───────────────────────────
    print("Building future curtailment features...")
    plant_futures = {}

    for plant_id in curt_feat_df["plant_id"].unique():
        plant_hist = curt_feat_df[curt_feat_df["plant_id"] == plant_id]
        plant_fc   = fc_df[fc_df["plant_id"] == plant_id]

        # Pull capacity and plant_type from the merged df (no separate master needed)
        cap   = float(df.loc[df["plant_id"] == plant_id, "capacity_mw"].iloc[0])
        ptype = str(df.loc[df["plant_id"] == plant_id, "plant_type"].iloc[0])

        if len(plant_fc) == 0:
            print(f"  [WARN] {plant_id} — no forecast rows found, skipping")
            continue

        try:
            future_feat = build_curtailment_future_features(
                plant_hist, plant_id, plant_fc, cap, ptype,
            )
            plant_futures[plant_id] = (future_feat, cap, ptype)
        except Exception as e:
            print(f"  [WARN] {plant_id} future features failed: {e}")

    # ── Build parallel tasks ──────────────────────────────────────────
    tasks = [
        (
            pid,
            curt_feat_df[curt_feat_df["plant_id"] == pid].copy(),
            plant_futures[pid][0],
            plant_futures[pid][1],
            plant_futures[pid][2],
            val_days,
        )
        for pid in plant_futures
    ]

    import multiprocessing
    max_cores     = multiprocessing.cpu_count()
    resolved_jobs = min(max_cores if n_jobs == -1 else n_jobs, len(tasks))

    print(f"\n{'═'*60}")
    print(f"  FLEET CURTAILMENT FORECAST")
    print(f"  Plants   : {len(tasks)}")
    print(f"  Workers  : {resolved_jobs} / {max_cores} cores")
    print(f"  Val days : {val_days}")
    print(f"{'═'*60}\n")

    # ── Parallel execution ────────────────────────────────────────────
    results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
        delayed(_process_curtailment_one_plant)(*task)
        for task in tasks
    )

    # ── Collect results ───────────────────────────────────────────────
    all_forecasts = []
    for res in results:
        if res["status"] == "ok" and res["forecast_df"] is not None:
            all_forecasts.append(res["forecast_df"])
            print(f"  ✓ {res['plant_id']}")
        else:
            print(f"  ✗ {res['plant_id']}: {res.get('error', '')[:80]}")

    if not all_forecasts:
        print("[WARN] No curtailment forecasts generated")
        return pd.DataFrame()

    combined = pd.concat(all_forecasts, ignore_index=True)
    combined.to_csv(OUT_DIR / "curtailment_forecasts.csv", index=False)
    print(f"\nCurtailment forecasts saved → {OUT_DIR}/curtailment_forecasts.csv")
    return combined