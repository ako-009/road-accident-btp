"""
etl.py  (Extract, Transform, Load)
------------------------------------
Step 1 of the pipeline.
Reads raw CSV -> validates columns -> cleans numbers -> saves clean version.
All other modules read from the cleaned CSV only.

Run with:
    python src/etl.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (RAW_DIR, CLEAN_CSV, DATA_DICT_CSV,
                        OUTPUT_DIR, ensure_dirs)

# Columns that must exist or we stop with a clear error
REQUIRED = [
    "state", "year", "total_accidents", "fatal_accidents",
    "killed", "grievous_injury", "minor_hosp",
]

# Columns that must be numbers
NUMERIC = [
    "year", "total_accidents", "fatal_accidents", "grievous_injury",
    "minor_hosp", "minor_no_hosp", "no_injury", "total_persons",
    "killed", "grievously_injured", "minor_injured",
    "night_accidents", "nh_accidents", "sh_accidents",
]


def run_etl():
    ensure_dirs()
    print("\n=== ETL Pipeline ===")

    # ── Load ──────────────────────────────────────────────────
    files = list(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV found in {RAW_DIR}. "
            "Run python src/make_synthetic_data.py first."
        )
    df = pd.read_csv(files[0], encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── Validate ──────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    print("  All required columns found.")

    # ── Clean numeric columns ─────────────────────────────────
    for col in NUMERIC:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            # Remove commas like "12,345" -> "12345"
            df[col] = df[col].str.replace(",", "", regex=False)
        df[col] = (pd.to_numeric(df[col], errors="coerce")
                   .fillna(0).astype(int).clip(lower=0))

    # ── Remove bad rows ───────────────────────────────────────
    n_before = len(df)
    df = df[df["state"].notna() & (df["state"].str.strip() != "")]
    df = df[(df["year"] >= 2000) & (df["year"] <= 2030)]
    df = df[df["total_accidents"] > 0]
    df = df.reset_index(drop=True)
    removed = n_before - len(df)
    if removed:
        print(f"  Removed {removed} invalid rows.")

    # ── Add derived columns ───────────────────────────────────
    safe = df["total_accidents"].replace(0, np.nan)

    # Deaths per 100 accidents — higher means more dangerous state
    df["fatality_rate"] = (df["killed"] / safe * 100).round(2).fillna(0)

    # Combined severity: (deaths + grievous) / accidents
    df["severity_index"] = (
        (df["killed"] + df["grievous_injury"]) / safe
    ).round(4).fillna(0)

    # Time period label for grouping in charts
    df["period"] = df["year"].apply(
        lambda y: "Pre-COVID" if y <= 2019
        else "COVID" if y <= 2021
        else "Post-COVID"
    )

    # ── Save cleaned CSV ──────────────────────────────────────
    df.to_csv(CLEAN_CSV, index=False)
    print(f"  Saved {len(df)} clean rows  ->  {CLEAN_CSV}")

    # ── Save data dictionary ──────────────────────────────────
    dd = pd.DataFrame([{
        "column":  c,
        "dtype":   str(df[c].dtype),
        "missing": int(df[c].isna().sum()),
        "min":     df[c].min() if df[c].dtype != object else "N/A",
        "max":     df[c].max() if df[c].dtype != object else "N/A",
    } for c in df.columns])
    dd.to_csv(DATA_DICT_CSV, index=False)
    print(f"  Data dictionary  ->  {DATA_DICT_CSV}")

    print(f"\n=== ETL complete. {len(df)} rows ready for analysis. ===")
    return df


if __name__ == "__main__":
    df = run_etl()
    print(f"\nTop 5 states by accidents in 2022:")
    yr   = df[df["year"] == 2022]
    top5 = yr.nlargest(5, "total_accidents")
    print(top5[["state", "total_accidents", "killed", "fatality_rate"]]
          .to_string(index=False))
