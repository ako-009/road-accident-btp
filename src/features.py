"""
features.py
-----------
Reads the cleaned CSV and creates new columns (features)
that are better suited for the Poisson regression model.

Feature engineering = turning raw numbers into smarter inputs.

Run with:
    python src/features.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import CLEAN_CSV, OUTPUT_DIR, ensure_dirs

FEATURES_CSV = OUTPUT_DIR / "features.csv"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the following new features:
    - log_accidents  : log of accident count (handles skewed distribution)
    - log_killed     : log of deaths
    - is_covid_year  : 1 for 2020-2021, else 0
    - is_large_state : 1 if state consistently has >5000 accidents/year
    - nh_share       : fraction of accidents on national highways
    - night_share    : fraction of accidents at night
    - yoy_change     : year-over-year % change in accidents per state
    """
    df = df.copy()

    # Log transform: accident counts are right-skewed,
    # log makes them more normally distributed which helps the model
    df["log_accidents"] = np.log1p(df["total_accidents"])
    df["log_killed"]    = np.log1p(df["killed"])

    # COVID flag: roads were empty in 2020-2021
    df["is_covid_year"] = df["year"].apply(lambda y: 1 if y in [2020, 2021] else 0)

    # Large state flag: states with high baseline accident counts
    state_avg = (df.groupby("state")["total_accidents"]
                   .mean()
                   .rename("state_avg_accidents"))
    df = df.merge(state_avg, on="state", how="left")
    threshold = df["state_avg_accidents"].median()
    df["is_large_state"] = (df["state_avg_accidents"] > threshold).astype(int)

    # Share of accidents on national highways (0 to 1)
    safe_acc = df["total_accidents"].replace(0, np.nan)
    df["nh_share"]    = (df["nh_accidents"]    / safe_acc).round(4).fillna(0)
    df["night_share"] = (df["night_accidents"] / safe_acc).round(4).fillna(0)

    # Year-over-year % change in accidents for each state
    df = df.sort_values(["state", "year"])
    df["prev_accidents"] = df.groupby("state")["total_accidents"].shift(1)
    df["yoy_change"] = (
        (df["total_accidents"] - df["prev_accidents"])
        / df["prev_accidents"].replace(0, np.nan) * 100
    ).round(2).fillna(0)

    # Drop helper column
    df = df.drop(columns=["prev_accidents", "state_avg_accidents"])

    return df.reset_index(drop=True)


def run_features() -> pd.DataFrame:
    ensure_dirs()
    print("\n=== Feature Engineering ===")

    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"Cleaned CSV not found. Run python src/etl.py first."
        )

    df = pd.read_csv(CLEAN_CSV)
    print(f"  Loaded {len(df)} rows")

    df = build_features(df)
    df.to_csv(FEATURES_CSV, index=False)

    new_cols = ["log_accidents", "log_killed", "is_covid_year",
                "is_large_state", "nh_share", "night_share", "yoy_change"]
    print(f"  Added {len(new_cols)} new feature columns:")
    for c in new_cols:
        print(f"    {c}")
    print(f"  Saved {len(df)} rows  ->  {FEATURES_CSV}")
    print("\n=== Feature engineering complete. ===")
    return df


if __name__ == "__main__":
    df = run_features()
    print("\nSample (5 rows):")
    cols = ["state", "year", "total_accidents", "log_accidents",
            "fatality_rate", "is_covid_year", "nh_share"]
    print(df[cols].head().to_string(index=False))
