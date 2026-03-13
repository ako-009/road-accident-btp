"""
make_synthetic_data.py
----------------------
Generates realistic fake road accident data for all Indian states.
This lets you run the full pipeline even without real iRAD data.
When you get real data later, just drop it in data/raw/ and nothing else changes.

Run with:
    python src/make_synthetic_data.py
"""

import random
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DIR, ensure_dirs

# Fixed seed = same numbers every time = reproducible research
random.seed(42)
np.random.seed(42)

STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Chandigarh",
    "Jammu & Kashmir", "Ladakh", "Puducherry", "Dadra & Nagar Haveli",
    "Daman & Diu", "Lakshadweep",
]

# Bigger states get proportionally more accidents
SCALE = {
    "Uttar Pradesh": 2.3, "Maharashtra": 1.8, "Tamil Nadu": 1.6,
    "Madhya Pradesh": 1.5, "Rajasthan": 1.4, "Karnataka": 1.4,
    "West Bengal": 1.3, "Gujarat": 1.3, "Andhra Pradesh": 1.2,
    "Telangana": 1.1, "Bihar": 1.1, "Kerala": 1.0, "Delhi": 1.0,
}

CAUSES = [
    "Over Speeding", "Drunk Driving", "Jumping Red Light",
    "Use of Mobile Phone", "Lane Violation", "Road Condition",
    "Vehicle Defect", "Other",
]

YEARS = [2019, 2020, 2021, 2022, 2023]


def make_row(state, year):
    """Generates one row of accident statistics for a state and year."""
    s    = SCALE.get(state, 0.5)
    base = max(100, int(np.random.normal(12000 * s, 2000 * s)))

    # COVID year had fewer vehicles on road = fewer accidents
    if year == 2020:
        base = int(base * 0.72)

    fatal    = int(base * np.random.uniform(0.28, 0.33))
    killed   = int(fatal * np.random.uniform(1.05, 1.2))
    grievous = int(base * np.random.uniform(0.30, 0.34))
    minor_h  = int(base * np.random.uniform(0.23, 0.27))
    minor_nh = int(base * np.random.uniform(0.04, 0.06))
    persons  = int(base * np.random.uniform(2.0, 2.5))

    return {
        "state":               state,
        "year":                year,
        "total_accidents":     base,
        "fatal_accidents":     fatal,
        "grievous_injury":     grievous,
        "minor_hosp":          minor_h,
        "minor_no_hosp":       minor_nh,
        "no_injury":           max(0, base - fatal - grievous - minor_h - minor_nh),
        "total_persons":       persons,
        "killed":              killed,
        "grievously_injured":  int(persons * np.random.uniform(0.18, 0.22)),
        "minor_injured":       int(persons * np.random.uniform(0.30, 0.36)),
        "night_accidents":     int(base * np.random.uniform(0.35, 0.45)),
        "nh_accidents":        int(base * np.random.uniform(0.30, 0.40)),
        "sh_accidents":        int(base * np.random.uniform(0.20, 0.28)),
        "top_cause":           random.choice(CAUSES[:5]),
        "two_wheeler_pct":     round(np.random.uniform(0.38, 0.52), 3),
        "pedestrian_pct":      round(np.random.uniform(0.10, 0.18), 3),
    }


def main():
    ensure_dirs()
    print("Generating synthetic road accident data...")
    rows = [make_row(s, y) for s in STATES for y in YEARS]
    df   = pd.DataFrame(rows)
    df["fatality_rate"] = (df["killed"] / df["total_accidents"] * 100).round(2)
    df   = df.sort_values(["year", "state"]).reset_index(drop=True)
    out  = RAW_DIR / "accidents_india.csv"
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} rows  ->  {out}")
    print(f"  States : {df['state'].nunique()}")
    print(f"  Years  : {sorted(df['year'].unique())}")
    print("Done! Now run python src/etl.py")


if __name__ == "__main__":
    main()
