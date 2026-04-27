"""
Karnataka Renewable Energy Grid — Feature Engineering Pipeline
==============================================================
Consolidated from:
  - solar_feature_engineering()
  - wind_feature_engineering()
  - plant_behavior_features()

Duplicates removed:
  - hour, day_of_year       (was in solar + wind → moved to shared base)
  - gen_lag_1/24/168        (was in wind + plant_behavior → kept in plant_behavior only)
  - cuf                     (was in wind + plant_behavior → kept in plant_behavior only)
  - wind_speed_rolling_std  (wind) merged with gen_rolling_std (plant_behavior)

Usage:
  df = build_features(df, plant_type="Solar",  capacity_kw=50_000)
  df = build_features(df, plant_type="Wind",   capacity_kw=80_000)
  df = build_features(df, plant_type="Hybrid", capacity_kw=100_000)
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — SHARED BASE FEATURES
# Applied to ALL plant types (Solar, Wind, Hybrid)
# ══════════════════════════════════════════════════════════════════════════════

def _base_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time features shared by every plant type.
    Extracted here so solar/wind functions don't duplicate them.
    """
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["month"]       = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding — avoids the 23→0 hour discontinuity
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]        / 24)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2A — SOLAR-SPECIFIC FEATURES
# Requires: irradiance, temperature, cloud_cover, generation
# ══════════════════════════════════════════════════════════════════════════════

def _solar_features(df: pd.DataFrame, capacity_kw: float) -> pd.DataFrame:
    """
    Solar physics + panel efficiency features.
    NOTE: hour/day_of_year NOT repeated here (handled by _base_time_features).
    """
    epsilon = 1e-6

    # 1. Clear-sky irradiance approximation (sinusoidal)
    #    pvlib can replace this for production use
    df["clear_sky_irradiance"] = (
        1000 * np.sin(np.pi * df["hour"] / 24)
    ).clip(lower=0)

    # 2. Cloud-adjusted irradiance
    df["irradiance_adjusted"] = df["irradiance"] * (1 - df["cloud_cover"] / 100)

    # 3. Irradiance ratio — actual vs theoretical clear sky
    df["irradiance_ratio"] = (
        df["irradiance"] / (df["clear_sky_irradiance"] + epsilon)
    ).clip(0, 1.5)

    # 4. Temperature efficiency loss (~0.4 %/°C above 25°C)
    df["temp_effect"] = 1 + (-0.004) * (df["temperature"] - 25)

    # 5. Expected generation (physics estimate)
    df["expected_generation"] = (
        df["irradiance_adjusted"] * df["temp_effect"] * capacity_kw / 1000
    )

    # 6. Performance Ratio — actual vs irradiance-weighted capacity
    df["performance_ratio"] = (
        df["generation"] / ((df["irradiance"] + epsilon) * capacity_kw)
    ).clip(0, 1.2)

    # 7. Rolling PR (panel degradation signal)
    df["pr_rolling_7"]  = df["performance_ratio"].rolling(7).mean()
    df["pr_rolling_30"] = df["performance_ratio"].rolling(30).mean()

    # 8. Soiling loss (dust accumulation — resets on cleaning events)
    df["days_since_cleaning"] = (
        df["timestamp"] - df["timestamp"].min()
    ).dt.days
    df["soiling_loss"] = (1 - 0.001 * df["days_since_cleaning"]).clip(0.70, 1.0)

    # 9. Combined adjusted generation signal
    df["adjusted_generation_signal"] = (
        df["expected_generation"] * df["soiling_loss"]
    )

    # 10. Daytime flag
    df["is_daylight"] = (df["clear_sky_irradiance"] > 0).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2B — WIND-SPECIFIC FEATURES
# Requires: wind_speed, wind_direction, temperature, generation
# ══════════════════════════════════════════════════════════════════════════════

def _wind_features(
    df: pd.DataFrame,
    air_density: float = 1.225,
    rotor_diameter: float = 100,
) -> pd.DataFrame:
    """
    Wind physics + turbine curve features.
    NOTE: hour/day_of_year NOT repeated (handled by _base_time_features).
    NOTE: cuf, gen_lag_* NOT here (handled by _plant_behavior_features).
    """
    swept_area = np.pi * (rotor_diameter / 2) ** 2

    # 1. Wind speed polynomial transforms (power ∝ v³)
    df["wind_speed_squared"] = df["wind_speed"] ** 2
    df["wind_speed_cubed"]   = df["wind_speed"] ** 3   # dominant physics feature

    # 2. Wind power density  P = 0.5 · ρ · A · v³
    df["wind_power_density"] = (
        0.5 * air_density * swept_area * df["wind_speed_cubed"]
    ) / 1e6   # scale to avoid huge numbers

    # 3. Turbine power curve (cut-in 3 m/s → rated 12 m/s → cut-out 25 m/s)
    def _power_curve(v):
        return np.where(
            v < 3,  0,
            np.where(v <= 12, (v ** 3) / (12 ** 3),
            np.where(v <= 25, 1.0,
                     0))
        )
    df["turbine_efficiency"] = _power_curve(df["wind_speed"].values)

    # 4. Expected wind generation
    df["expected_wind_generation"] = (
        df["wind_power_density"] * df["turbine_efficiency"]
    )

    # 5. Wind direction → sin/cos (handles circular nature of degrees)
    df["wind_dir_rad"] = np.deg2rad(df["wind_direction"])
    df["wind_dir_sin"] = np.sin(df["wind_dir_rad"])
    df["wind_dir_cos"] = np.cos(df["wind_dir_rad"])
    df.drop(columns=["wind_dir_rad"], inplace=True)   # intermediate, not a feature

    # 6. Air density adjusted for temperature (density ↓ as temp ↑)
    df["air_density_adjusted"] = air_density * (273 / (273 + df["temperature"]))
    df["adjusted_wpd"] = (
        0.5 * df["air_density_adjusted"] * swept_area * df["wind_speed_cubed"]
    ) / 1e6

    # 7. Operational regime flags
    df["is_below_cut_in"]  = (df["wind_speed"] < 3).astype(int)
    df["is_above_cut_out"] = (df["wind_speed"] > 25).astype(int)

    # 8. Wind ramp (variability signal — NOT duplicated in plant_behavior)
    df["wind_speed_diff"]          = df["wind_speed"].diff()
    df["wind_speed_rolling_std_6"] = df["wind_speed"].rolling(6).std()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — PLANT BEHAVIOR FEATURES
# Applied to ALL plant types AFTER type-specific layers
# Contains: lags, rolling stats, CUF, ramp, downtime — previously duplicated
# ══════════════════════════════════════════════════════════════════════════════

def _plant_behavior_features(df: pd.DataFrame, capacity_kw: float) -> pd.DataFrame:
    """
    Temporal + behavioral features common to every plant type.
    This is the single source of: lags, CUF, ramp, rolling stats.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Lag features (temporal autocorrelation)
    df["gen_lag_1"]   = df["generation"].shift(1)    # t-1 hour
    df["gen_lag_24"]  = df["generation"].shift(24)   # same hour yesterday
    df["gen_lag_168"] = df["generation"].shift(168)  # same hour last week

    # 2. Rolling statistics (trend + volatility)
    df["gen_rolling_mean_6"]   = df["generation"].rolling(6).mean()
    df["gen_rolling_std_6"]    = df["generation"].rolling(6).std()
    df["gen_rolling_mean_24"]  = df["generation"].rolling(24).mean()
    df["gen_rolling_mean_168"] = df["generation"].rolling(168).mean()

    # 3. Capacity Utilization Factor (single source — removed from wind layer)
    df["cuf"] = (df["generation"] / (capacity_kw + 1e-6)).clip(0, 1.2)

    # 4. Ramp rate (sudden changes signal)
    df["ramp_rate"] = df["generation"].diff()
    df["ramp_abs"]  = df["ramp_rate"].abs()

    # 5. Momentum (short-term trend direction)
    df["gen_momentum_3"] = df["generation"] - df["generation"].shift(3)
    df["gen_momentum_6"] = df["generation"] - df["generation"].shift(6)

    # 6. Load factor (daily energy vs theoretical max)
    df["daily_generation"] = df["generation"].rolling(24).sum()
    df["load_factor"] = df["daily_generation"] / (capacity_kw * 24 + 1e-6)

    # 7. Weekly peak indicator
    rolling_max = df["generation"].rolling(168).max()
    df["is_peak"] = (df["generation"] >= 0.9 * rolling_max).astype(int)

    # 8. Generation variability (stability signal)
    df["gen_variability_24"]  = df["generation"].rolling(24).std()
    df["gen_variability_168"] = df["generation"].rolling(168).std()

    # 9. Downtime / zero-generation detection
    df["is_zero_gen"] = (df["generation"] == 0).astype(int)
    df["zero_streak"] = (
        df["is_zero_gen"]
        .groupby((df["is_zero_gen"] != df["is_zero_gen"].shift()).cumsum())
        .cumsum()
    )

    # 10. Residual from rolling expectation
    df["gen_residual_24"] = df["generation"] - df["gen_rolling_mean_24"]

    # 11. Normalized generation (cross-plant comparability)
    df["gen_normalized"] = df["generation"] / (df["gen_rolling_mean_168"] + 1e-6)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — Single entry point
# ══════════════════════════════════════════════════════════════════════════════

def build_features(
    df: pd.DataFrame,
    plant_type: str,
    air_density: float = 1.225,
    rotor_diameter: float = 100,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline for Solar, Wind, or Hybrid plants.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'timestamp' and 'generation'.
        Solar/Hybrid also need: irradiance, temperature, cloud_cover.
        Wind/Hybrid also need: wind_speed, wind_direction, temperature.

    plant_type : str
        One of: "Solar" | "Wind" | "Hybrid"

    capacity_kw : float
        Installed capacity in kW.

    air_density : float
        Air density kg/m³ (wind/hybrid only). Default 1.225.

    rotor_diameter : float
        Turbine rotor diameter in metres (wind/hybrid only). Default 100.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe, NaN rows dropped.

    Example
    -------
    >>> df_solar  = build_features(df, "Solar",  capacity_kw=50_000)
    >>> df_wind   = build_features(df, "Wind",   capacity_kw=80_000)
    >>> df_hybrid = build_features(df, "Hybrid", capacity_kw=100_000)
    """
    plant_type = plant_type.strip().title()
    assert plant_type in ("Solar", "Wind", "Hybrid"), (
        f"plant_type must be 'Solar', 'Wind', or 'Hybrid'. Got: {plant_type}"
    )

    df = df.copy()

    # ── Layer 1: shared time features (no duplication downstream) ──
    df = _base_time_features(df)

    # Derive per-row capacity in kW from the column that's already there
    # This works for both scalar lookup and vectorised operations
    capacity_kw_col = df["capacity_mw"] * 1000   # pd.Series, shape (n,)


    # ── Layer 2: plant-type-specific physics ──
    if plant_type == "Solar":
        df = _solar_features(df, capacity_kw_col)

    elif plant_type == "Wind":
        df = _wind_features(df, air_density, rotor_diameter)

    elif plant_type == "Hybrid":
        # Both solar AND wind features — no column collision since each
        # produces distinct feature names
        df = _solar_features(df, capacity_kw_col)
        df = _wind_features(df, air_density, rotor_diameter)

    # ── Layer 3: shared behavioral features (lags, rolling, CUF, etc.) ──
    df = _plant_behavior_features(df, capacity_kw_col)

    # Drop NaNs introduced by rolling/lag windows
    df = df.dropna().reset_index(drop=True)

    return df
