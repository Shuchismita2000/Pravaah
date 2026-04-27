"""
Karnataka Renewable Energy Grid — Pre-Processing Pipeline
=========================================================
Run this BEFORE feature_engineering.py

Input columns (from your merged dataframe):
    plant_id, plant_type, capacity_mw,
    actual_generation_mw, availability_mw, curtailment_mw,
    health_factor, irradiance_wm2,
    temperature, cloud_cover, wind_speed, wind_direction,
    irradiance,                         ← duplicate of irradiance_wm2, needs resolution
    year, month, day, hour              ← split time parts, need to be rebuilt into timestamp

Steps:
    1. Schema validation      — check required columns exist, right dtypes
    2. Duplicate column fix   — irradiance vs irradiance_wm2
    3. Physical bounds check  — clip impossible sensor values
    4. Missing value strategy — per-column imputation logic
    5. Outlier treatment      — IQR + domain-aware capping
    6. Categorical encoding   — plant_type → integer codes + dummies
    7. Derived base columns   — things FE pipeline expects but aren't in raw data
    8. Per-plant normalisation — optional, for cross-plant ML
    9. Final dtype cast       — everything to float32 except IDs
    10. QA report              — summary of what was changed
"""

import pandas as pd
import numpy as np
from typing import Tuple
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

def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check required columns exist. Warn on missing optional ones.
    Raises ValueError if any required column is absent.
    """
    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    optional = ["temperature", "cloud_cover", "wind_speed", "wind_direction", "irradiance"]
    missing_optional = [c for c in optional if c not in df.columns]
    if missing_optional:
        print(f"[WARN] Optional columns missing (will be filled with defaults): {missing_optional}")

    print(f"[OK] Schema validated — {len(df):,} rows, {len(df.columns)} columns")
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

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
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
        # Generation — forward fill short gaps, then 0 for long outages
        group["actual_generation_mw"] = (
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
# STEP 7 — CATEGORICAL ENCODING
# ══════════════════════════════════════════════════════════════════════════════

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    plant_type needs to be encoded for ML models.

    Two encodings added (let the modeller pick):
      - plant_type_code : integer ordinal  (0=Solar, 1=Wind, 2=Hybrid)
      - plant_type_*    : one-hot dummies  (better for tree models)

    plant_id is kept as string for grouping — do NOT encode it here,
    leave that for embedding or target-encoding in the model layer.
    """
    df["plant_type"] = df["plant_type"].str.strip().str.title()
    unmapped = df[~df["plant_type"].isin(PLANT_TYPE_MAP)]["plant_type"].unique()
    if len(unmapped) > 0:
        print(f"[WARN] Unknown plant_type values: {unmapped} — will be NaN in code column")

    df["plant_type_code"] = df["plant_type"].map(PLANT_TYPE_MAP)

    # One-hot (drop_first=False — keep all 3, model can select)
    dummies = pd.get_dummies(df["plant_type"], prefix="plant_type", dtype=int)
    df = pd.concat([df, dummies], axis=1)

    print(f"[OK] Encoded plant_type → codes + dummies: {dummies.columns.tolist()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — DERIVED BASE COLUMNS (FE PIPELINE PREREQUISITES)
# ══════════════════════════════════════════════════════════════════════════════

def add_derived_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    The feature engineering pipeline expects a column called 'generation'.
    Your raw data calls it 'actual_generation_mw'. Align here, not inside FE.

    Also adds a few simple derived columns that every downstream model needs
    but that don't belong in the FE physics layers.
    """
    # Rename for FE pipeline compatibility
    if "actual_generation_mw" in df.columns and "generation" not in df.columns:
        df["generation"] = df["actual_generation_mw"]
        print("[OK] Aliased actual_generation_mw → generation")

    # Capacity factor (actual / nameplate) — fundamental derived signal
    df["capacity_factor"] = (
        df["actual_generation_mw"] / (df["capacity_mw"] + 1e-6)
    ).clip(0, 1.2)

    # Generation shortfall vs availability (degradation signal)
    df["generation_shortfall_mw"] = (
        df["availability_mw"] - df["actual_generation_mw"]
    ).clip(lower=0)

    # Net availability after curtailment (what could theoretically be dispatched)
    df["net_availability_mw"] = (
        df["availability_mw"] - df["curtailment_mw"]
    ).clip(lower=0)

    # Health-adjusted capacity (theoretical max given machine state)
    df["health_adjusted_capacity_mw"] = df["capacity_mw"] * df["health_factor"]

    # Boolean: is plant in degraded state (health < 75%)
    df["is_degraded"] = (df["health_factor"] < 0.75).astype(int)

    # Boolean: is plant likely in repair/offline (health < 55%)
    df["is_offline"]  = (df["health_factor"] < 0.55).astype(int)

    print("[OK] Derived base columns added")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — OPTIONAL PER-PLANT NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalise_per_plant(
    df: pd.DataFrame,
    method: str = "capacity",   # "capacity" | "zscore" | "minmax" | "none"
) -> pd.DataFrame:
    """
    Cross-plant ML models need generation on a comparable scale.
    A 20 MW plant generating 15 MW is very different from a 200 MW plant
    generating 15 MW.

    method='capacity'  → divide by capacity_mw (default, most interpretable)
    method='zscore'    → per-plant z-score (mean=0, std=1)
    method='minmax'    → per-plant 0–1 scaling
    method='none'      → skip (if training per-plant models)
    """
    if method == "none":
        print("[SKIP] Per-plant normalisation skipped")
        return df

    if method == "capacity":
        df["generation_norm"] = df["generation"] / (df["capacity_mw"] + 1e-6)

    elif method == "zscore":
        stats = df.groupby("plant_id")["generation"].agg(["mean", "std"]).reset_index()
        stats.columns = ["plant_id", "gen_mean", "gen_std"]
        df = df.merge(stats, on="plant_id", how="left")
        df["generation_norm"] = (df["generation"] - df["gen_mean"]) / (df["gen_std"] + 1e-6)
        df.drop(columns=["gen_mean", "gen_std"], inplace=True)

    elif method == "minmax":
        stats = df.groupby("plant_id")["generation"].agg(["min", "max"]).reset_index()
        stats.columns = ["plant_id", "gen_min", "gen_max"]
        df = df.merge(stats, on="plant_id", how="left")
        df["generation_norm"] = (
            (df["generation"] - df["gen_min"]) / (df["gen_max"] - df["gen_min"] + 1e-6)
        )
        df.drop(columns=["gen_min", "gen_max"], inplace=True)

    print(f"[OK] Per-plant normalisation: {method}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — FINAL DTYPE OPTIMISATION
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
# STEP 11 — QA REPORT
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
    key_cols = ["actual_generation_mw", "capacity_factor", "health_factor",
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
    normalise_method: str = "capacity",
    run_qa: bool = True,
) -> pd.DataFrame:
    """
    Full pre-processing pipeline. Run this before build_features().

    Parameters
    ----------
    df               : Raw merged dataframe (generation + weather joined)
    outlier_method   : "iqr" | "zscore" | "none"
    normalise_method : "capacity" | "zscore" | "minmax" | "none"
    run_qa           : Print QA report at end

    Returns
    -------
    pd.DataFrame : Clean, transformed dataframe ready for feature engineering

    Example
    -------
    >>> df_clean = preprocess(df_raw)
    >>> df_features = build_features(df_clean, plant_type="Solar", capacity_kw=50_000)
    """
    print("\n" + "─"*60)
    print("  STARTING PRE-PROCESSING PIPELINE")
    print("─"*60)

    df = validate_schema(df)
    df = resolve_irradiance(df)
    df = cast_numeric(df)
    df = enforce_physical_bounds(df)
    df = impute_missing(df)
    df = treat_outliers(df, method=outlier_method)
    df = encode_categoricals(df)
    df = add_derived_base_columns(df)
    df = normalise_per_plant(df, method=normalise_method)
    df = optimise_dtypes(df)

    if run_qa:
        qa_report(df)

    print("  PRE-PROCESSING COMPLETE")
    print("─"*60 + "\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pandas as pd

    # ── Load raw data ──
    gen     = pd.read_csv("data/generation_raw.csv")
    weather = pd.read_csv("data/weather_raw.csv")
    pm      = pd.read_csv("data/plant_master.csv")

    # ── Join generation + weather on timestamp + region ──
    # (match plant to its regional weather zone)
    gen = gen.merge(pm[["plant_id", "region"]], on="plant_id", how="left")
    gen = gen.merge(
        weather[["timestamp", "region", "temperature_c", "cloud_cover_pct",
                 "wind_speed_kmh", "wind_direction_deg"]],
        on=["timestamp", "region"],
        how="left",
    ).rename(columns={
        "temperature_c":    "temperature",
        "cloud_cover_pct":  "cloud_cover",
        "wind_speed_kmh":   "wind_speed",
        "wind_direction_deg": "wind_direction",
    })

    # ── Split timestamp → year/month/day/hour (as your schema has them) ──
    gen["timestamp"] = pd.to_datetime(gen["timestamp"])
    gen["year"]  = gen["timestamp"].dt.year
    gen["month"] = gen["timestamp"].dt.month
    gen["day"]   = gen["timestamp"].dt.day
    gen["hour"]  = gen["timestamp"].dt.hour

    # ── Run preprocessing ──
    df_clean = preprocess(gen)

    print("Columns after preprocessing:")
    print(df_clean.columns.tolist())
    print(f"\nReady for feature engineering: {df_clean.shape}")

    # ── Now pass to FE pipeline ──
    # from feature_engineering import build_features
    # df_solar = df_clean[df_clean["plant_type"]=="Solar"].copy()
    # df_fe = build_features(df_solar, "Solar", capacity_kw=50_000)