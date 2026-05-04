"""
Step A — Univariate generation forecasting
Purpose: produce 72-hour generation forecasts per plant
         so those values can feed into the multivariate pipeline as lagged features.

Models compared per plant:
  - SARIMA   (captures seasonality, no external regressors)
  - Prophet  (handles multiple seasonalities + holidays cleanly)
  - LSTM     (learns non-linear temporal patterns)

Output: best_model per plant + 72-hour forecast array
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn

from joblib import Parallel, delayed
import os
import traceback
from pathlib import Path

OUT_DIR = Path("data/forecasts")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — Data Pre-processing
# Important to Run this to make the univariate models work well — they expect clean, regular, gap-free hourly series
# ══════════════════════════════════════════════════════════════════════════════
def prepare_univariate_input(
    df: pd.DataFrame,
    min_hours: int = 500,
) -> pd.DataFrame:
    """
    Prepares a clean, minimal dataframe for univariate forecasting.

    Input  : df with at minimum — timestamp, plant_id, 
             plant_type, actual_generation_mw
    Output : df with timestamp, plant_id, plant_type, generation
             — sorted, hourly-complete, no gaps, no negatives
    """


    # ── 1. Types ──────────────────────────────────────────────────
    df['timestamp']     = pd.to_datetime(df['timestamp'])
    df['actual_generation_mw'] = pd.to_numeric(df['actual_generation_mw'], errors='coerce')

    # ── 2. Rename to match what univariate expects ─────────────────
    # run_univariate_fleet looks for column named 'generation'
    df = df.rename(columns={'actual_generation_mw': 'generation'})

    # ── 3. Remove duplicates — keep last (most recent SCADA write) ─
    before = len(df)
    df = df.drop_duplicates(subset=['plant_id', 'timestamp'], keep='last')
    dropped = before - len(df)
    if dropped:
        print(f"[Prep] Dropped {dropped:,} duplicate timestamp rows")

    # ── 4. Sort ───────────────────────────────────────────────────
    df = df.sort_values(['plant_id', 'timestamp']).reset_index(drop=True)

    # ── 5. Per-plant: enforce hourly frequency + fill gaps ─────────
    # This is the most important step — models break on irregular timestamps
    parts = []
    skipped = []

    for plant_id, plant_df in df.groupby('plant_id'):

        plant_type = plant_df['plant_type'].iloc[0]

        series = (
            plant_df
            .set_index('timestamp')['generation']
            .sort_index()
        )

        # Enforce complete hourly index — no missing hours
        full_idx = pd.date_range(
            start = series.index.min(),
            end   = series.index.max(),
            freq  = 'h',
        )
        series = series.reindex(full_idx)

        n_gaps = series.isna().sum()

        # Fill strategy — same as univariate runner does internally
        # but doing it here makes gaps visible before modelling
        series = series.ffill(limit=3).fillna(0)

        if n_gaps:
            print(f"  [{plant_id}] Filled {n_gaps:,} missing hours "
                  f"({n_gaps/len(series):.1%} of series)")

        # ── 6. Physical bounds ────────────────────────────────────
        # Negative generation = sensor error, clip to 0
        n_neg = (series < 0).sum()
        if n_neg:
            print(f"  [{plant_id}] Clipped {n_neg} negative values to 0")
        series = series.clip(lower=0)

        # ── 7. Skip plants with insufficient data ─────────────────
        if len(series) < min_hours:
            print(f"  [{plant_id}] SKIPPED — only {len(series)} hrs "
                  f"(need {min_hours})")
            skipped.append(plant_id)
            continue

        # Rebuild as dataframe
        plant_out = pd.DataFrame({
            'timestamp':  series.index,
            'plant_id':   plant_id,
            'plant_type': plant_type,
            'generation': series.values,
        })
        parts.append(plant_out)

    # ── 8. Combine ────────────────────────────────────────────────
    if not parts:
        raise ValueError("No plants passed the minimum hours filter. "
                         f"Check min_hours={min_hours} or your data.")

    out = pd.concat(parts, ignore_index=True)

    # ── 9. Summary ────────────────────────────────────────────────
    print(f"\n[Prep] Ready for univariate forecasting:")
    print(f"  Plants included : {out['plant_id'].nunique()}")
    print(f"  Plants skipped  : {len(skipped)}")
    print(f"  Total rows      : {len(out):,}")
    print(f"  Columns         : {out.columns.tolist()}")
    print(f"  Date range      : {out['timestamp'].min()} → {out['timestamp'].max()}")
    print(f"\n  Rows per plant type:")
    print(out.groupby('plant_type')['plant_id'].nunique()
            .rename('plants').to_string())

    return out

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TIME SERIES DIAGNOSTICS
# Run this first per plant to understand the series before choosing a model
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_series(series: pd.Series, plant_id: str) -> dict:
    """
    Stationarity, seasonality strength, and missing rate.
    Tells you which model family is most appropriate.

    Returns a dict of diagnostics used to auto-select model.
    """
    series = series.dropna()

    # 1. Stationarity — ADF test
    adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
    is_stationary = adf_p < 0.05

    # 2. Seasonal strength — variance ratio at lag 24 (daily) and lag 168 (weekly)
    acf_vals = acf(series, nlags=170, fft=True)
    daily_strength  = abs(acf_vals[24])
    weekly_strength = abs(acf_vals[168])

    # 3. Zero rate (hours with no generation — nights for solar)
    zero_rate = (series == 0).mean()

    # 4. Coefficient of variation (how volatile)
    cv = series.std() / (series.mean() + 1e-6)

    diag = {
        "plant_id":        plant_id,
        "n_obs":           len(series),
        "is_stationary":   is_stationary,
        "adf_pvalue":      round(adf_p, 4),
        "daily_acf":       round(daily_strength, 3),
        "weekly_acf":      round(weekly_strength, 3),
        "zero_rate":       round(zero_rate, 3),
        "cv":              round(cv, 3),
        "mean_mw":         round(series.mean(), 2),
        "recommended_model": _recommend_model(is_stationary, daily_strength,
                                               weekly_strength, zero_rate, cv),
    }
    print(f"[{plant_id}] stationary={is_stationary} | daily_acf={daily_strength:.2f} "
          f"| zero_rate={zero_rate:.1%} | CV={cv:.2f} → {diag['recommended_model']}")
    return diag


def _recommend_model(stationary, daily, weekly, zero_rate, cv):
    # High zero rate = solar plant with clear day/night pattern → Prophet handles this best
    if zero_rate > 0.35:
        return "prophet"
    # Strong weekly seasonality + non-stationary → SARIMA with seasonal differencing
    if weekly > 0.3 and not stationary:
        return "sarima"
    # High volatility or complex non-linear pattern → LSTM
    if cv > 1.2:
        return "lstm"
    # Default: Prophet (robust, handles missing, multiple seasonalities)
    return "prophet"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TRAIN / VALIDATION SPLIT
# Always split on time, never shuffle for time series
# ══════════════════════════════════════════════════════════════════════════════

def time_split(series: pd.Series, val_hours: int = 168):
    """
    Last `val_hours` hours for validation (default = 7 days).
    Everything before = training.
    """
    train = series.iloc[:-val_hours]
    val   = series.iloc[-val_hours:]
    return train, val


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # MAPE — skip zeros (common for solar at night)
    mask = actual > 0.1
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.sum() else np.nan
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE": round(mape, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SARIMA
# Good for: wind plants, stationary or near-stationary series
# ══════════════════════════════════════════════════════════════════════════════

def fit_sarima(train: pd.Series, horizon: int = 72) -> tuple:
    """
    SARIMA(1,1,1)(1,1,1,24) — daily seasonality.
    Auto-selects d based on stationarity test.

    Returns: (model_fit, forecast_array)
    """
    # Determine differencing order
    _, p_val, *_ = adfuller(train.dropna())
    d = 0 if p_val < 0.05 else 1

    model = SARIMAX(
        train,
        order=(1, d, 1),
        seasonal_order=(1, 1, 1, 24),  # 24-hour daily seasonality
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    forecast = fit.forecast(steps=horizon)
    forecast = np.maximum(0, forecast)  # generation can't be negative

    print(f"  SARIMA AIC={fit.aic:.1f} | BIC={fit.bic:.1f}")
    return fit, forecast.values


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PROPHET
# Good for: solar plants, multiple seasonalities, missing data, zero-heavy series
# ══════════════════════════════════════════════════════════════════════════════

KARNATAKA_HOLIDAYS = pd.DataFrame({
    "holiday": ["Republic Day", "Independence Day", "Gandhi Jayanti", "Karnataka Rajyotsava"] * 3,
    "ds": pd.to_datetime([
        "2023-01-26", "2023-08-15", "2023-10-02", "2023-11-01",
        "2024-01-26", "2024-08-15", "2024-10-02", "2024-11-01",
        "2025-01-26", "2025-08-15", "2025-10-02", "2025-11-01",
    ]),
})


def fit_prophet(train: pd.Series, horizon: int = 72) -> tuple:
    """
    Prophet with daily + weekly + yearly seasonality and Karnataka holidays.
    Handles the zero-at-night pattern for solar naturally.

    Returns: (model, forecast_df with 'yhat' column)
    """
    df = train.reset_index()
    df.columns = ["ds", "y"]

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        holidays=KARNATAKA_HOLIDAYS,
        seasonality_mode="multiplicative",  # better for solar (scales with capacity)
        changepoint_prior_scale=0.05,        # conservative — don't overfit recent changes
        interval_width=0.90,
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=horizon, freq="h")
    forecast = model.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)

    future_only = forecast.tail(horizon)
    print(f"  Prophet fitted — {horizon}h forecast range: "
          f"[{future_only['yhat'].min():.1f}, {future_only['yhat'].max():.1f}] MW")
    return model, future_only


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LSTM
# Good for: hybrid plants, high volatility, complex non-linear patterns
# ══════════════════════════════════════════════════════════════════════════════

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=72):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # take last timestep output


def fit_lstm(
    train: pd.Series,
    horizon: int = 72,
    lookback: int = 168,   # 7 days of hourly history as input window
    epochs: int = 50,
    batch_size: int = 32,
) -> tuple:
    """
    LSTM trained on sliding windows of `lookback` hours to predict next `horizon` hours.

    Returns: (model, scaler, forecast_array)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Scale to [0, 1]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()

    # Build supervised dataset: X = window of lookback, y = next horizon
    X_list, y_list = [], []
    for i in range(len(scaled) - lookback - horizon + 1):
        X_list.append(scaled[i : i + lookback])
        y_list.append(scaled[i + lookback : i + lookback + horizon])

    X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1).to(device)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).to(device)

    model = LSTMForecaster(output_size=horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X, y)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  LSTM epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.5f}")

    # Forecast: use last `lookback` hours of training data as seed
    model.eval()
    with torch.no_grad():
        seed = torch.tensor(scaled[-lookback:], dtype=torch.float32)
        seed = seed.unsqueeze(0).unsqueeze(-1).to(device)
        pred_scaled = model(seed).cpu().numpy().flatten()

    forecast = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    forecast = np.maximum(0, forecast)
    return model, scaler, forecast


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Other Models (ETS, Theta, LightGBM, XGBoost)
# ETS (Exponential Smoothing, near-instant, good baseline)
# Theta  (fast classical, strong on energy M3/M4 benchmarks)
# XGBoost  (alternative to LightGBM, ~15s)
# LightGBM  (replaces LSTM — 8s vs 12min)
# ══════════════════════════════════════════════════════════════════════════════


try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[WARN] lightgbm not installed")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.forecasting.theta import ThetaModel
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ══════════════════════════════════════════════════════════════════════════
# SHARED UTILITY — lag feature builder for tree models
# ══════════════════════════════════════════════════════════════════════════

def _build_lag_features(series: pd.Series) -> pd.DataFrame:
    """
    Converts a univariate series into a supervised learning dataframe
    using lag features. This is what makes tree models work for time series.

    Features created:
      - Lags: t-1, t-2, t-3, t-6, t-12, t-24, t-48, t-168
      - Rolling: mean/std over 6h, 24h, 168h windows
      - Time: hour, day_of_week, month, is_weekend, is_daytime
    """
    df = pd.DataFrame({"y": series})

    # Lag features — most important for tree models
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # Rolling statistics
    for window in [6, 24, 168]:
        df[f"roll_mean_{window}"] = df["y"].shift(1).rolling(window).mean()
        df[f"roll_std_{window}"]  = df["y"].shift(1).rolling(window).std()
        df[f"roll_max_{window}"]  = df["y"].shift(1).rolling(window).max()

    # Calendar features from index
    df["hour"]        = series.index.hour
    df["day_of_week"] = series.index.dayofweek
    df["month"]       = series.index.month
    df["is_weekend"]  = (series.index.dayofweek >= 5).astype(int)
    df["is_daytime"]  = ((series.index.hour >= 6) & (series.index.hour <= 18)).astype(int)

    # Cyclical encoding — avoids 23→0 discontinuity
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.dropna()
    return df


def _tree_train_predict(
    series: pd.Series,
    model_obj,
    horizon: int,
) -> np.ndarray:
    """
    Trains a tree model (LightGBM or XGBoost) on lag features,
    then forecasts horizon steps ahead using recursive prediction.

    Recursive strategy: predict t+1, append to series, predict t+2, etc.
    This is the standard approach for multi-step tree forecasting.
    """
    # Build features on full training series
    feat_df = _build_lag_features(series)
    feature_cols = [c for c in feat_df.columns if c != "y"]

    X_train = feat_df[feature_cols].values
    y_train = feat_df["y"].values

    model_obj.fit(X_train, y_train)

    # Recursive multi-step forecast
    history   = series.copy()
    forecasts = []

    for step in range(horizon):
        # Rebuild features on growing history, take last row
        feat = _build_lag_features(history)
        if len(feat) == 0:
            forecasts.append(history.mean())
            continue

        x_pred = feat[feature_cols].iloc[[-1]].values
        pred   = float(model_obj.predict(x_pred)[0])
        pred   = max(0, pred)      # generation >= 0
        forecasts.append(pred)

        # Append prediction to history for next step
        next_ts    = history.index[-1] + pd.Timedelta(hours=1)
        new_point  = pd.Series([pred], index=[next_ts])
        history    = pd.concat([history, new_point])

    return np.array(forecasts)
# ══════════════════════════════════════════════════════════════════════════════
# ETS (Exponential Smoothing, near-instant, good baseline)
# Theta  (fast classical, strong on energy M3/M4 benchmarks)
# XGBoost  (alternative to LightGBM, ~15s)
# LightGBM  (replaces LSTM — 8s vs 12min)
# ══════════════════════════════════════════════════════════════════════════════

def fit_ets(
    train: pd.Series,
    horizon: int,
) -> tuple[object, np.ndarray]:
    """
    Holt-Winters Exponential Smoothing with additive trend + seasonality.
    Fits in < 2s. Strong on data with clear daily pattern.
    Good replacement for naive baseline.
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels not installed")

    # ETS needs at least 2 full seasonal cycles
    if len(train) < 48:
        raise ValueError("ETS needs at least 48 observations (2 × 24h cycles)")

    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=24,    # daily cycle
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True, remove_bias=True)
    fc  = fit.forecast(horizon)
    fc  = np.maximum(0, fc.values)

    print(f"    ETS: alpha={fit.params['smoothing_level']:.3f} | "
          f"forecast range [{fc.min():.1f}, {fc.max():.1f}] MW")
    return fit, fc

def fit_theta(
    train: pd.Series,
    horizon: int,
) -> tuple[object, np.ndarray]:
    """
    Theta method — decomposes series into trend + seasonality.
    Extremely fast (< 1s), competitive on energy data.
    Good fallback when SARIMA/Prophet are too slow.
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels not installed")

    # Theta requires positive values — clip zeros to small positive
    train_pos = train.clip(lower=0.01)

    model = ThetaModel(
        train_pos,
        period=24,          # daily seasonality
        deseasonalize=True,
        use_test=False,
    )
    fit = model.fit(disp=False)
    fc  = fit.forecast(horizon)
    fc  = np.maximum(0, fc.values)

    print(f"    Theta: forecast range [{fc.min():.1f}, {fc.max():.1f}] MW")
    return fit, fc

def fit_xgboost(
    train: pd.Series,
    horizon: int,
) -> tuple[object, np.ndarray]:
    """
    XGBoost on lag features. Slightly slower than LightGBM but often
    more robust on smaller datasets with higher regularisation.
    """
    if not HAS_XGB:
        raise ImportError("xgboost not installed: pip install xgboost")

    feat_df      = _build_lag_features(train)
    feature_cols = [c for c in feat_df.columns if c != "y"]

    X = feat_df[feature_cols].values
    y = feat_df["y"].values

    split       = int(len(X) * 0.85)
    X_tr, X_vl = X[:split], X[split:]
    y_tr, y_vl = y[:split], y[split:]

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        n_jobs=1,
        early_stopping_rounds=50,
        eval_metric="mae",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

    fc = _tree_train_predict(train, model, horizon)
    print(f"    XGBoost: {model.best_iteration} trees | "
          f"forecast range [{fc.min():.1f}, {fc.max():.1f}] MW")
    return model, fc


def fit_lightgbm(
    train: pd.Series,
    horizon: int,
) -> tuple[object, np.ndarray]:
    """
    LightGBM on lag features. Trains in seconds, competitive accuracy.

    Key params:
      n_estimators=500  — enough trees, early stopping prevents overfit
      num_leaves=31     — default, good for small-medium datasets
      learning_rate=0.05 — conservative, more stable than 0.1
    """
    if not HAS_LGBM:
        raise ImportError("lightgbm not installed: pip install lightgbm")

    feat_df      = _build_lag_features(train)
    feature_cols = [c for c in feat_df.columns if c != "y"]

    X = feat_df[feature_cols].values
    y = feat_df["y"].values

    # Time-based train/val split for early stopping (no shuffle)
    split    = int(len(X) * 0.85)
    X_tr, X_vl = X[:split], X[split:]
    y_tr, y_vl = y[:split], y[split:]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1,
        n_jobs=1,           # 1 here — parallelism handled at fleet level
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    fc = _tree_train_predict(train, model, horizon)
    print(f"    LightGBM: {model.best_iteration_} trees | "
          f"forecast range [{fc.min():.1f}, {fc.max():.1f}] MW")
    return model, fc

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MODEL SELECTION: run all, pick best on validation MAE
# ══════════════════════════════════════════════════════════════════════════════

def select_best_univariate_model(
    series: pd.Series,
    plant_id: str,
    plant_type: str,            # "solar" | "wind" | "hybrid"
    horizon: int = 72,
    val_hours: int = 168,
    lstm_mae_threshold: float = 15.0,  # MW — tune per plant/fleet
) -> dict:
    """
    Model roster (in order of typical speed):
      ETS       ~1s    — exponential smoothing baseline
      Theta     ~1s    — classical decomposition
      SARIMA    ~30s   — statistical, strong for wind
      Prophet   ~60s   — handles solar zero-at-night
      LightGBM  ~8s    — gradient boosting on lags, often best overall
      XGBoost   ~15s   — alternative to LightGBM
      LSTM      ~12min — hybrid plants only, and only when base models
                         fail to beat lstm_mae_threshold

    LSTM gate logic:
      - solar / wind  → LSTM never runs (saves ~12 min per plant)
      - hybrid        → LSTM runs only when best base MAE > lstm_mae_threshold
                        i.e. treat it as an expensive fallback, not the default

    plant_type is used ONLY for this gate — it is not fed as a feature
    into any model to avoid data leakage.
    """
    _VALID_PLANT_TYPES = {"solar", "wind", "hybrid"}
    if plant_type not in _VALID_PLANT_TYPES:
        raise ValueError(
            f"plant_type must be one of {_VALID_PLANT_TYPES}, got '{plant_type}'"
        )

    diag       = diagnose_series(series, plant_id)
    train, val = time_split(series, val_hours)
    results    = {}

    # ── ETS ──────────────────────────────────────────────────────────
    try:
        _, ets_fc = fit_ets(train, horizon=val_hours)
        results["ets"] = {"scores": evaluate(val.values, ets_fc[:val_hours])}
    except Exception as e:
        print(f"  [ETS failed] {e}")

    # ── Theta ─────────────────────────────────────────────────────────
    try:
        _, theta_fc = fit_theta(train, horizon=val_hours)
        results["theta"] = {"scores": evaluate(val.values, theta_fc[:val_hours])}
    except Exception as e:
        print(f"  [Theta failed] {e}")

    # ── SARIMA ────────────────────────────────────────────────────────
    try:
        _, sarima_fc = fit_sarima(train, horizon=val_hours)
        results["sarima"] = {"scores": evaluate(val.values, sarima_fc[:val_hours])}
    except Exception as e:
        print(f"  [SARIMA failed] {e}")

    # ── Prophet ───────────────────────────────────────────────────────
    try:
        _, prophet_fc = fit_prophet(train, horizon=val_hours)
        results["prophet"] = {"scores": evaluate(val.values, prophet_fc["yhat"].values)}
    except Exception as e:
        print(f"  [Prophet failed] {e}")

    # ── LightGBM ─────────────────────────────────────────────────────
    try:
        _, lgbm_fc = fit_lightgbm(train, horizon=val_hours)
        results["lightgbm"] = {"scores": evaluate(val.values, lgbm_fc[:val_hours])}
    except Exception as e:
        print(f"  [LightGBM failed] {e}")

    # ── XGBoost ───────────────────────────────────────────────────────
    try:
        _, xgb_fc = fit_xgboost(train, horizon=val_hours)
        results["xgboost"] = {"scores": evaluate(val.values, xgb_fc[:val_hours])}
    except Exception as e:
        print(f"  [XGBoost failed] {e}")

    if not results:
        raise RuntimeError(f"All base models failed for {plant_id}")

    # ── LSTM: hybrid-only, threshold-gated fallback ───────────────────
    best_base_mae = min(r["scores"]["MAE"] for r in results.values())

    _run_lstm = (
        plant_type == "hybrid"
        and best_base_mae > lstm_mae_threshold
    )

    if _run_lstm:
        print(
            f"  [LSTM] Triggered — plant_type=hybrid, "
            f"best base MAE={best_base_mae:.3f} > threshold={lstm_mae_threshold}"
        )
        try:
            _, _, lstm_fc = fit_lstm(train, horizon=val_hours)
            results["lstm"] = {"scores": evaluate(val.values, lstm_fc[:val_hours])}
        except Exception as e:
            print(f"  [LSTM failed] {e}")
    elif plant_type == "hybrid":
        print(
            f"  [LSTM] Skipped — base models sufficient "
            f"(best MAE={best_base_mae:.3f} ≤ threshold={lstm_mae_threshold})"
        )
    else:
        print(f"  [LSTM] Skipped — plant_type={plant_type} (hybrid only)")

    # ── Pick winner on MAE ────────────────────────────────────────────
    best_name = min(results, key=lambda m: results[m]["scores"]["MAE"])

    print(f"\n  Winner for {plant_id} ({plant_type}): {best_name.upper()}")
    for m, r in results.items():
        flag = " <--" if m == best_name else ""
        print(
            f"    {m:10s}: MAE={r['scores']['MAE']:.3f}  "
            f"RMSE={r['scores']['RMSE']:.3f}  "
            f"MAPE={r['scores']['MAPE']:.1f}%{flag}"
        )

    # ── Refit winner on FULL series ───────────────────────────────────
    print(f"\n  Refitting {best_name} on full series for {horizon}h forecast...")

    forecast_timestamps = pd.date_range(
        start=series.index[-1] + pd.Timedelta(hours=1),
        periods=horizon, freq="h",
    )
    fc_mw    = np.zeros(horizon)
    fc_lower = np.zeros(horizon)
    fc_upper = np.zeros(horizon)

    if best_name == "ets":
        fit_obj, fc_mw = fit_ets(series, horizon=horizon)
        fc_lower = np.maximum(0, fc_mw * 0.85)
        fc_upper = fc_mw * 1.15

    elif best_name == "theta":
        fit_obj, fc_mw = fit_theta(series, horizon=horizon)
        fc_lower = np.maximum(0, fc_mw * 0.85)
        fc_upper = fc_mw * 1.15

    elif best_name == "sarima":
        fit_obj, fc_mw = fit_sarima(series, horizon=horizon)
        try:
            ci       = fit_obj.get_forecast(steps=horizon).conf_int(alpha=0.10)
            fc_lower = np.maximum(0, ci.iloc[:, 0].values)
            fc_upper = ci.iloc[:, 1].values
        except Exception:
            fc_lower = np.maximum(0, fc_mw * 0.85)
            fc_upper = fc_mw * 1.15

    elif best_name == "prophet":
        _, fc_df_full = fit_prophet(series, horizon=horizon)
        fc_mw    = fc_df_full["yhat"].values
        fc_lower = fc_df_full["yhat_lower"].clip(lower=0).values
        fc_upper = fc_df_full["yhat_upper"].values

    elif best_name == "lightgbm":
        fit_obj, fc_mw = fit_lightgbm(series, horizon=horizon)
        rmse     = results["lightgbm"]["scores"]["RMSE"]
        fc_lower = np.maximum(0, fc_mw - 1.5 * rmse)
        fc_upper = fc_mw + 1.5 * rmse

    elif best_name == "xgboost":
        fit_obj, fc_mw = fit_xgboost(series, horizon=horizon)
        rmse     = results["xgboost"]["scores"]["RMSE"]
        fc_lower = np.maximum(0, fc_mw - 1.5 * rmse)
        fc_upper = fc_mw + 1.5 * rmse

    elif best_name == "lstm":
        _, _, fc_mw = fit_lstm(series, horizon=horizon)
        rmse     = results["lstm"]["scores"]["RMSE"]
        fc_lower = np.maximum(0, fc_mw - 1.5 * rmse)
        fc_upper = fc_mw + 1.5 * rmse

    forecast_df = pd.DataFrame({
        "plant_id":    plant_id,
        "timestamp":   forecast_timestamps,
        "forecast_mw": np.round(fc_mw,    3),
        "lower_90":    np.round(fc_lower, 3),
        "upper_90":    np.round(fc_upper, 3),
        "model":       best_name,
    })

    return {
        "plant_id":            plant_id,
        "plant_type":          plant_type,
        "best_model":          best_name,
        "model_object":        fit_obj,
        "lstm_triggered":      _run_lstm,
        "diagnostics":         diag,
        "all_scores":          {m: r["scores"] for m, r in results.items()},
        "forecast_72h":        fc_mw,
        "forecast_timestamps": forecast_timestamps,
        "forecast_df":         forecast_df,
    }
# ══════════════════════════════════════════════════════════════════════════
# WORKER — runs in its own process, one plant at a time
# Must be a top-level function (not a lambda/nested) for joblib to pickle it
# ══════════════════════════════════════════════════════════════════════════
import joblib
from pathlib import Path
from datetime import datetime

MODEL_DIR = Path("models/generation")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def _process_one_plant(
    plant_id: str,
    plant_df: pd.DataFrame,
    plant_type: str,            # ← added
    horizon: int,
    lstm_mae_threshold: float,  # ← added (pass-through to model selector)
) -> dict:
    """
    Isolated worker function called by joblib for each plant.

    Returns a result dict with keys:
        status       : "ok" | "skip" | "error"
        plant_id     : str
        plant_type   : str
        forecast_df  : pd.DataFrame | None
        log_row      : dict | None
        error        : str | None
    """
    warnings.filterwarnings("ignore")

    try:
        # ── Build series ──────────────────────────────────────────────
        series = (
            plant_df
            .sort_values("timestamp")
            .set_index("timestamp")["generation"]
            .asfreq("h")
            .ffill(limit=3)
            .fillna(0)
        )

        if len(series) < 500:
            return {
                "status":      "skip",
                "plant_id":    plant_id,
                "plant_type":  plant_type,
                "forecast_df": None,
                "log_row":     None,
                "error":       f"insufficient_data: {len(series)} hrs",
            }

        # ── Model selection ───────────────────────────────────────────
        result = select_best_univariate_model(
            series,
            plant_id,
            plant_type=plant_type,                  # ← new
            horizon=horizon,
            lstm_mae_threshold=lstm_mae_threshold,  # ← new
        )
        model_obj   = result["model_object"]
        model_name  = result["best_model"]

        
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        model_path = MODEL_DIR / f"{plant_id}_generation_model_{ts}.joblib"

        joblib.dump(
            {
            "model": model_obj,
            "plant_id": plant_id,
            "plant_type": plant_type,
            "model_name": model_name,
        },
        model_path,
        compress=3,
    )

        fc_df = result.get("forecast_df")
        if fc_df is None or len(fc_df) == 0:
            return {
                "status":      "error",
                "plant_id":    plant_id,
                "plant_type":  plant_type,
                "forecast_df": None,
                "log_row":     None,
                "error":       "empty forecast_df returned",
            }

        log_row = {
            "plant_id":       plant_id,
            "plant_type":     plant_type,           # ← new (useful in model_selection_log.csv)
            "n_features":     len(plant_df.columns),
            "n_obs":          result["diagnostics"]["n_obs"],
            "zero_rate":      result["diagnostics"]["zero_rate"],
            "cv":             result["diagnostics"]["cv"],
            "best_model":     result["best_model"],
            "lstm_triggered": result["lstm_triggered"],  # ← new (track LSTM usage in log)
        }
        for model_name, sc in result["all_scores"].items():
            log_row[f"{model_name}_MAE"]  = sc["MAE"]
            log_row[f"{model_name}_RMSE"] = sc["RMSE"]
            log_row[f"{model_name}_MAPE"] = sc["MAPE"]

        return {
            "status":      "ok",
            "plant_id":    plant_id,
            "plant_type":  plant_type,
            "forecast_df": fc_df,
            "log_row":     log_row,
            "error":       None,
        }

    except Exception as e:
        return {
            "status":      "error",
            "plant_id":    plant_id,
            "plant_type":  plant_type,
            "forecast_df": None,
            "log_row":     None,
            "error":       f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }



# ══════════════════════════════════════════════════════════════════════════
# FLEET RUNNER — parallelised with joblib
# ═════════════════════════════════════════════════════════════════════════=

def run_univariate_fleet(
    fleet_df: pd.DataFrame,
    horizon: int = 72,
    output_path: str = "data/forecasts/univariate_forecasts.csv",
    n_jobs: int = -1,
    backend: str = "loky",
    lstm_mae_threshold: float = 15.0,   # ← added; forwarded to every worker
) -> pd.DataFrame:
    """
    Parallelised univariate model selection and forecasting for all plants.

    Parameters
    ----------
    fleet_df    : Pre-processed dataframe, all plants.
                  Required columns: plant_id, timestamp, generation, plant_type.
                  plant_type must be one of: "solar" | "wind" | "hybrid"

    horizon     : Forecast horizon in hours. Default 72.

    output_path : Where to save the combined forecast CSV.

    n_jobs      : Number of parallel workers.
                  -1  = use all CPU cores          (fastest, high RAM)
                   1  = sequential, no parallelism  (easiest to debug)
                   4  = use 4 cores                 (balanced)

                  Rule of thumb for this workload:
                    - SARIMA + Prophet are CPU-bound → more cores = faster
                    - LSTM is GPU-bound if CUDA available → set n_jobs=1
                      and let LSTM use the full GPU per plant sequentially
                    - If no GPU: n_jobs = min(cpu_count, n_plants // 2)

    backend     : joblib backend.
                  "loky"      = default, process-based, safest for statsmodels/prophet
                  "threading" = thread-based, use only if models release the GIL

    lstm_mae_threshold : MAE ceiling (MW) above which LSTM is tried on hybrid
                         plants. Has no effect on solar/wind. Default 15.0.

    Returns
    -------
    pd.DataFrame : Combined forecast for all plants (long format).
    """
    # ── Validate columns ──────────────────────────────────────────────
    required = {"plant_id", "timestamp", "generation", "plant_type"}  # ← plant_type added
    missing  = required - set(fleet_df.columns)
    if missing:
        raise ValueError(
            f"fleet_df missing required columns: {missing}\n"
            f"Ensure plant_type ('solar'|'wind'|'hybrid') is present before calling."
        )

    # ── Validate plant_type values ────────────────────────────────────
    valid_types   = {"Solar", "Wind", "Hybrid"}
    bad_types     = set(fleet_df["plant_type"].unique()) - valid_types
    if bad_types:
        raise ValueError(
            f"Unknown plant_type values found: {bad_types}. "
            f"Must be one of {valid_types}."
        )

    fleet_df = fleet_df.copy()
    fleet_df["timestamp"] = pd.to_datetime(fleet_df["timestamp"])
    fleet_df["plant_type"] = fleet_df["plant_type"].str.strip().str.lower() 
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Split into per-plant dataframes ───────────────────────────────
    # Pull plant_type once per plant — it's constant within a plant group
    plant_groups = [
        (plant_id, plant_df.reset_index(drop=True), plant_df["plant_type"].iloc[0])
        for plant_id, plant_df in fleet_df.groupby("plant_id")
    ]
    n_plants = len(plant_groups)

    # ── Resolve n_jobs ────────────────────────────────────────────────
    import multiprocessing
    max_cores     = multiprocessing.cpu_count()
    resolved_jobs = max_cores if n_jobs == -1 else min(n_jobs, max_cores)
    resolved_jobs = min(resolved_jobs, n_plants)

    # ── Fleet composition summary ─────────────────────────────────────
    type_counts = fleet_df.groupby("plant_type")["plant_id"].nunique()

    print(f"\n{'═'*60}")
    print(f"  FLEET UNIVARIATE FORECAST")
    print(f"  Plants   : {n_plants}  "
          f"(solar={type_counts.get('Solar', 0)}  "
          f"wind={type_counts.get('Wind', 0)}  "
          f"hybrid={type_counts.get('Hybrid', 0)})")
    print(f"  Horizon  : {horizon}h")
    print(f"  Workers  : {resolved_jobs} / {max_cores} cores  (n_jobs={n_jobs})")
    print(f"  Backend  : {backend}")
    print(f"  LSTM threshold : {lstm_mae_threshold} MAE (hybrid only)")
    print(f"{'═'*60}\n")

    # ── Parallel execution ────────────────────────────────────────────
    results = Parallel(n_jobs=resolved_jobs, backend=backend, verbose=10)(
        delayed(_process_one_plant)(plant_id, plant_df, plant_type, horizon, lstm_mae_threshold)
        for plant_id, plant_df, plant_type in plant_groups  # ← unpack plant_type
    )

    # ── Collect results ───────────────────────────────────────────────
    all_forecasts = []
    selection_log = []
    failed_plants = []

    for res in results:
        pid   = res["plant_id"]
        ptype = res["plant_type"]
        if res["status"] == "ok":
            all_forecasts.append(res["forecast_df"])
            selection_log.append(res["log_row"])
            lstm_flag = " [LSTM]" if res["log_row"].get("lstm_triggered") else ""
            print(f"  ✓ {pid} ({ptype}) — {res['log_row']['best_model']}{lstm_flag}"
                  f"  MAE={res['log_row'].get(res['log_row']['best_model']+'_MAE', '?')}")
        elif res["status"] == "skip":
            print(f"  ⊘ {pid} ({ptype}) — skipped: {res['error']}")
            failed_plants.append({"plant_id": pid, "plant_type": ptype, "error": res["error"]})
        else:
            print(f"  ✗ {pid} ({ptype}) — ERROR: {res['error'].splitlines()[0]}")
            failed_plants.append({"plant_id": pid, "plant_type": ptype, "error": res["error"]})

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  FLEET COMPLETE")
    print(f"  Successful : {len(all_forecasts)} / {n_plants} plants")
    print(f"  Failed     : {len(failed_plants)} plants")

    log_df = pd.DataFrame(selection_log)
    if len(log_df):
        print(f"\n  Model distribution:")
        for model, cnt in log_df["best_model"].value_counts().items():
            print(f"    {model:10s}: {cnt:3d} plants ({cnt/len(log_df):.0%})")

        lstm_hits = log_df["lstm_triggered"].sum()
        if lstm_hits:
            print(f"\n  LSTM triggered on {lstm_hits} hybrid plant(s)")
    print(f"{'═'*60}\n")

    # ── Save logs ─────────────────────────────────────────────────────
    if len(log_df):
        log_df.to_csv(OUT_DIR / "model_selection_log.csv", index=False)

    if failed_plants:
        pd.DataFrame(failed_plants).to_csv(OUT_DIR / "failed_plants.csv", index=False)
        print(f"  Failed details saved → {OUT_DIR}/failed_plants.csv")

    # ── Guard concat ──────────────────────────────────────────────────
    if not all_forecasts:
        print("\n[FATAL] No forecasts generated.")
        print("  → Set n_jobs=1 to see full error output per plant")
        print("  → Check failed_plants.csv for error details")
        return pd.DataFrame()

    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    forecast_df.to_csv(output_path, index=False)
    print(f"  Forecast saved → {output_path}")
    return forecast_df