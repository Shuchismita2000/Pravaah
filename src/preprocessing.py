"""
Karnataka Renewable Energy Grid — Pre-Processing Pipeline
=========================================================

Steps:
    1. Schema validation      — check required columns exist, right dtypes
    2. Duplicate column fix   — irradiance vs irradiance_wm2
    3. Physical bounds check  — clip impossible sensor values
    4. Missing value strategy — per-column imputation logic
    5. Outlier treatment      — IQR + domain-aware capping
    6. Final dtype cast       — everything to float32 except IDs
    7. QA report              — summary of what was changed
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  — domain knowledge baked in
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_COLS = [
    "plant_id", "plant_type", "capacity_mw",
    "actual_generation_mw", "availability_mw", "curtailment_mw",
    "health_factor", "irradiance_wm2",
    "timestamp"
]

PHYSICAL_BOUNDS = {
    # column              : (min,    max)
    "actual_generation_mw": (0,      5000),   # no plant > 5 GW
    "availability_mw":      (0,      5000),
    "curtailment_mw":       (0,      5000),
    "capacity_mw":          (1,      5000),
    "health_factor":        (0,      1.0),    # 0 = dead, 1 = brand new
    "irradiance_wm2":       (0,      1400),   # theoretical solar max ~1361 W/m²
    "irradiance":           (0,      1400),
    "temperature":          (-10,    55),     # Karnataka range: 10–45°C realistic
    "cloud_cover":          (0,      100),    # percentage
    "wind_speed":           (0,      60),     # m/s — above 60 = sensor error
    "wind_direction":       (0,      360),
}

# Columns that must be non-negative (generation cannot be negative)
NON_NEGATIVE = [
    "actual_generation_mw", "availability_mw", "curtailment_mw",
    "capacity_mw", "health_factor", "irradiance_wm2", "irradiance",
    "cloud_cover", "wind_speed",
]

# Expected dtypes before casting to float32
NUMERIC_COLS = [
    "capacity_mw", "actual_generation_mw", "availability_mw",
    "curtailment_mw", "health_factor", "irradiance_wm2",
    "temperature", "cloud_cover", "wind_speed", "wind_direction",
]

PLANT_TYPE_MAP = {"Solar": 0, "Wind": 1, "Hybrid": 2}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SCHEMA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_schema(df: pd.DataFrame, is_forecast: bool = False) -> pd.DataFrame:
    
    required = [
        "timestamp", "plant_id", "capacity_mw",
        "temperature", "cloud_cover", "wind_speed",
        "wind_direction", "irradiance", "health_factor",
    ]

    # Only required during training, not at inference
    if is_forecast == False:
        required.append("actual_generation_mw")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — RESOLVE DUPLICATE IRRADIANCE COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def resolve_irradiance(df: pd.DataFrame) -> pd.DataFrame:
    """
    You have BOTH 'irradiance_wm2' and 'irradiance' — same physical quantity,
    likely from two different joins (generation_raw vs weather_raw).

    Strategy:
      - If both present: take the non-null value; where both present, average them.
      - Rename the winner to 'irradiance' (what FE pipeline expects).
      - Drop 'irradiance_wm2' to avoid downstream confusion.
    """
    if "irradiance_wm2" in df.columns and "irradiance" in df.columns:
        # How similar are they?
        both_valid = df["irradiance_wm2"].notna() & df["irradiance"].notna()
        if both_valid.sum() > 0:
            corr = df.loc[both_valid, ["irradiance_wm2", "irradiance"]].corr().iloc[0, 1]
            print(f"[INFO] irradiance_wm2 vs irradiance correlation: {corr:.4f}")

        # Merge: prefer irradiance_wm2 (from generation_raw, plant-specific)
        # fall back to irradiance (from weather_raw, zone-level)
        df["irradiance"] = df["irradiance_wm2"].combine_first(df["irradiance"])
        df.drop(columns=["irradiance_wm2"], inplace=True)
        print("[OK] Merged irradiance_wm2 + irradiance → single 'irradiance' column")

    elif "irradiance_wm2" in df.columns:
        df.rename(columns={"irradiance_wm2": "irradiance"}, inplace=True)
        print("[OK] Renamed irradiance_wm2 → irradiance")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FORCE NUMERIC DTYPES
# ══════════════════════════════════════════════════════════════════════════════

def cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Columns loaded from CSV may be object/string dtype.
    Force all numeric columns to float. Non-parseable → NaN (handled later).
    """
    for col in NUMERIC_COLS:
        if col in df.columns:
            before_nulls = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after_nulls = df[col].isna().sum()
            new_nulls = after_nulls - before_nulls
            if new_nulls > 0:
                print(f"[WARN] {col}: {new_nulls} values couldn't be parsed → NaN")

    # irradiance may have been renamed already
    if "irradiance" in df.columns:
        df["irradiance"] = pd.to_numeric(df["irradiance"], errors="coerce")

    print("[OK] Numeric dtypes enforced")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — PHYSICAL BOUNDS ENFORCEMENT
# ══════════════════════════════════════════════════════════════════════════════

def enforce_physical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sensor errors, SCADA glitches, and data entry mistakes can push values
    outside physically possible ranges. Cap them rather than drop — we lose
    less data and the clipping itself is informative.

    Also: generation cannot exceed availability (physics constraint).
    """
    clip_counts = {}
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        if out_of_range > 0:
            df[col] = df[col].clip(lo, hi)
            clip_counts[col] = int(out_of_range)

    if clip_counts:
        print(f"[OK] Physical bounds clipped: {clip_counts}")

    # Domain constraint: actual_generation ≤ availability_mw
    if "actual_generation_mw" in df.columns and "availability_mw" in df.columns:
        violation = (df["actual_generation_mw"] > df["availability_mw"]).sum()
        if violation > 0:
            df["actual_generation_mw"] = df[["actual_generation_mw", "availability_mw"]].min(axis=1)
            print(f"[OK] Fixed {violation} rows where generation > availability")

    # Domain constraint: actual_generation ≤ capacity_mw
    if "actual_generation_mw" in df.columns and "capacity_mw" in df.columns:
        violation2 = (df["actual_generation_mw"] > df["capacity_mw"]).sum()
        if violation2 > 0:
            df["actual_generation_mw"] = df[["actual_generation_mw", "capacity_mw"]].min(axis=1)
            print(f"[OK] Fixed {violation2} rows where generation > capacity")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — MISSING VALUE IMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def impute_missing(df: pd.DataFrame, is_forecast: bool = False) -> pd.DataFrame:
    """
    Different columns need different imputation strategies:

    | Column                  | Strategy                          | Why                         |
    |-------------------------|-----------------------------------|-----------------------------|
    | actual_generation_mw    | Forward fill → 0                  | Outage = 0, not interpolated|
    | availability_mw         | Forward fill within plant          | Slow-changing signal        |
    | curtailment_mw          | Fill 0                             | Missing = no curtailment    |
    | health_factor           | Forward fill within plant          | Monotone degradation        |
    | irradiance              | Interpolate (time), then 0         | Smooth physical signal      |
    | temperature             | Interpolate (time)                 | Slow-changing               |
    | cloud_cover             | Forward fill                       | Synoptic-scale persistence  |
    | wind_speed              | Interpolate (time)                 | Smooth physical signal      |
    | wind_direction          | Forward fill                       | Circular, ffill safest      |
    """
    def _per_plant(group):

        if is_forecast == False:
            # Generation — forward fill short gaps, then 0 for long outages
            group["generation"] = (
                group["actual_generation_mw"]
                .ffill(limit=3)   # fill up to 3 consecutive missing hours
                .fillna(0)
            )
        # Availability
        group["availability_mw"] = (
            group["availability_mw"]
            .ffill(limit=6)
            .bfill(limit=2)
            .fillna(group["capacity_mw"] * group["health_factor"])  # physics fallback
        )
        # Curtailment — missing = 0 (no curtailment event recorded)
        group["curtailment_mw"] = group["curtailment_mw"].fillna(0)

        # Health factor — monotone degradation, so forward fill is correct
        group["health_factor"] = (
            group["health_factor"]
            .ffill()
            .bfill()
            .fillna(0.85)   # global default if still missing
        )
        # Irradiance — smooth physical signal, interpolate is safe
        if "irradiance" in group.columns:
            group["irradiance"] = (
                group["irradiance"]
                .interpolate(method="linear", limit=6)
                .fillna(0)
            )
        # Temperature — slow-changing, interpolate safely
        if "temperature" in group.columns:
            group["temperature"] = (
                group["temperature"]
                .interpolate(method="linear", limit=12)
                .fillna(group["temperature"].median())
            )
        # Cloud cover
        if "cloud_cover" in group.columns:
            group["cloud_cover"] = (
                group["cloud_cover"]
                .ffill(limit=6)
                .fillna(30)   # moderate cloud as default
            )
        # Wind speed
        if "wind_speed" in group.columns:
            group["wind_speed"] = (
                group["wind_speed"]
                .interpolate(method="linear", limit=6)
                .fillna(5.0)   # ~median wind speed for Karnataka
            )
        # Wind direction — circular, don't interpolate
        if "wind_direction" in group.columns:
            group["wind_direction"] = (
                group["wind_direction"]
                .ffill(limit=3)
                .fillna(180)   # south wind as default
            )
        return group

    before_nulls = df.isnull().sum().sum()
    result_parts = []
    for pid in df["plant_id"].unique():
        mask = df["plant_id"] == pid
        result_parts.append(_per_plant(df.loc[mask].copy()))
    df = pd.concat(result_parts, ignore_index=True)
    after_nulls = df.isnull().sum().sum()
    print(f"[OK] Imputation complete — nulls reduced: {before_nulls:,} → {after_nulls:,}")

    if after_nulls > 0:
        remaining = df.isnull().sum()
        remaining = remaining[remaining > 0]
        print(f"[WARN] Remaining nulls:\n{remaining}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — OUTLIER TREATMENT
# ══════════════════════════════════════════════════════════════════════════════

def treat_outliers(df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
    """
    Outlier treatment per numeric column, applied per plant_type group
    (a 50 MW solar plant's 'outlier' is normal for a 150 MW plant).

    method='iqr'    → Winsorise at 1.5×IQR  (default, robust)
    method='zscore' → Cap at ±3 std dev      (assumes normality)
    method='none'   → Skip (if bounds already handled it)

    Columns EXCLUDED from outlier treatment:
      - capacity_mw     (fixed, not time-varying)
      - health_factor   (bounded 0–1 by physics)
      - curtailment_mw  (legitimate spikes)
      - wind_direction  (circular)
    """
    SKIP_COLS = {"capacity_mw", "health_factor", "curtailment_mw", "wind_direction",
                 "year", "month", "day", "hour", "plant_id", "plant_type", "timestamp"}

    target_cols = [
        c for c in NUMERIC_COLS + ["irradiance"]
        if c in df.columns and c not in SKIP_COLS
    ]

    if method == "none":
        print("[SKIP] Outlier treatment skipped")
        return df

    def _winsorise_group(group):
        for col in target_cols:
            if col not in group.columns:
                continue
            vals = group[col].dropna()
            if len(vals) < 10:
                continue
            if method == "iqr":
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            else:  # zscore
                mu, sigma = vals.mean(), vals.std()
                lo, hi = mu - 3 * sigma, mu + 3 * sigma
            # Never let bounds go below physical minimums
            lo = max(lo, PHYSICAL_BOUNDS.get(col, (-np.inf, np.inf))[0])
            hi = min(hi, PHYSICAL_BOUNDS.get(col, (-np.inf, np.inf))[1])
            group[col] = group[col].clip(lo, hi)
        return group

    # Apply winsorisation per plant_type without groupby (avoids column-drop bug)
    for ptype in df["plant_type"].unique():
        mask = df["plant_type"] == ptype
        subset = df.loc[mask].copy()
        subset = _winsorise_group(subset)
        df.loc[mask, target_cols] = subset[target_cols].values
    print(f"[OK] Outlier treatment done ({method}) on: {target_cols}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — FINAL DTYPE OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast numeric columns to float32 (half the memory of float64).
    Keep string/datetime columns as-is.
    Cast integer flag columns to int8 (saves 75% vs int64).

    With ~3M rows and 60+ features this saves ~1–2 GB RAM.
    """
    KEEP_AS_IS = {"plant_id", "plant_type", "timestamp"}
    INT8_COLS  = {
        "is_degraded", "is_offline", "is_weekend",
        "plant_type_code", "plant_type_Solar", "plant_type_Wind", "plant_type_Hybrid",
    }

    for col in df.columns:
        if col in KEEP_AS_IS:
            continue
        if col in INT8_COLS and col in df.columns:
            df[col] = df[col].astype("int8")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("int32")

    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"[OK] Dtypes optimised — memory usage: {mem_mb:.1f} MB")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — QA REPORT
# ══════════════════════════════════════════════════════════════════════════════

def qa_report(df: pd.DataFrame) -> None:
    """Print a concise data quality summary after preprocessing."""
    print("\n" + "═"*60)
    print("  PRE-PROCESSING QA REPORT")
    print("═"*60)
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Plants         : {df['plant_id'].nunique()}")
    print(f"  Plant types    : {df['plant_type'].value_counts().to_dict()}")
    print(f"  Date range     : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Null values    : {df.isnull().sum().sum()}")
    print(f"  Duplicate rows : {df.duplicated().sum()}")
    print(f"\n  Numeric summary:")
    key_cols = ["generation", "capacity_factor", "health_factor",
                "irradiance", "temperature", "wind_speed"]
    key_cols = [c for c in key_cols if c in df.columns]
    print(df[key_cols].describe().round(3).to_string())
    print("═"*60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — Single entry point
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(
    df: pd.DataFrame,
    outlier_method: str = "iqr",
    run_qa: bool = True,
    is_forecast: bool = False,     
) -> pd.DataFrame:
    """
    Full pre-processing pipeline. Run this before build_features().

    Parameters
    ----------
    df               : Raw merged dataframe (generation + weather joined)
    outlier_method   : "iqr" | "zscore" | "none"
    normalise_method : "capacity" | "zscore" | "minmax" | "none"
    run_qa           : Print QA report at end
    is_forecast      : If True, drops generation before processing
                       (use when passing forecast_df at inference time)

    Returns
    -------
    pd.DataFrame : Clean, transformed dataframe ready for feature engineering
    """
    print("\n" + "─"*60)
    print("  STARTING PRE-PROCESSING PIPELINE")
    print("─"*60)

    df = validate_schema(df, is_forecast=is_forecast)
    df = resolve_irradiance(df)
    df = cast_numeric(df)
    df = enforce_physical_bounds(df)
    df = impute_missing(df, is_forecast=is_forecast)
    df = treat_outliers(df, method=outlier_method)
    df = optimise_dtypes(df)

    if run_qa:
        qa_report(df)

    print("  PRE-PROCESSING COMPLETE")
    print("─"*60 + "\n")
    return df