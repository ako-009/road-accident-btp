"""
blackspot.py
-------------
Identifies road accident black spots — states or regions where
accidents and deaths are disproportionately high.

This mirrors the iRAD black spot logic (from your PDF):
  A black spot = location with 5+ accidents OR 10+ fatalities
  in a 500m stretch over 3 calendar years.

Since we have state-level data (not GPS-level), we define a
state as a black spot if its fatality rate OR accident count
is in the top 25% nationally.

Run with:
    python src/blackspot.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (CLEAN_CSV, BLACKSPOTS_CSV,
                        PLOTS_DIR, OUTPUT_DIR, ensure_dirs)

YEAR_SHOW = 2022


def identify_blackspots(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Flags states as black spots based on three criteria:
      1. Fatality rate in top 25% nationally
      2. Total accidents in top 25% nationally
      3. Severity index in top 25% nationally

    A state is a black spot if it meets ANY two of the three criteria.
    """
    yr = df[df["year"] == year].copy()

    # Calculate thresholds (75th percentile = top 25%)
    thresh_fatality  = yr["fatality_rate"].quantile(0.75)
    thresh_accidents = yr["total_accidents"].quantile(0.75)
    thresh_severity  = yr["severity_index"].quantile(0.75)

    # Flag each criterion
    yr["flag_fatality"]  = (yr["fatality_rate"]  >= thresh_fatality).astype(int)
    yr["flag_accidents"] = (yr["total_accidents"] >= thresh_accidents).astype(int)
    yr["flag_severity"]  = (yr["severity_index"]  >= thresh_severity).astype(int)

    # Total flags per state (0, 1, 2, or 3)
    yr["flags_total"] = (yr["flag_fatality"] +
                         yr["flag_accidents"] +
                         yr["flag_severity"])

    # Black spot if 2 or more flags triggered
    yr["is_blackspot"] = (yr["flags_total"] >= 2).astype(int)

    # Risk level label
    yr["risk_level"] = yr["flags_total"].map({
        0: "Low",
        1: "Medium",
        2: "High",
        3: "Critical",
    })

    result = yr[[
        "state", "year", "total_accidents", "killed",
        "fatality_rate", "severity_index",
        "flag_fatality", "flag_accidents", "flag_severity",
        "flags_total", "is_blackspot", "risk_level"
    ]].sort_values("flags_total", ascending=False).reset_index(drop=True)

    return result


def recommend_interventions(row: pd.Series) -> str:
    """
    Maps black spot criteria to evidence-based interventions
    from the Campbell Systematic Review (Sections 7-8).
    These are the 4E interventions: Engineering, Enforcement,
    Education, Emergency.
    """
    interventions = []

    if row["flag_fatality"]:
        # High deaths -> better emergency response + road engineering
        interventions.append(
            "Engineering: Install crash barriers, improve road geometry at junctions"
        )
        interventions.append(
            "Emergency: Reduce ambulance response time, position CATS ambulances at hotspots"
        )

    if row["flag_accidents"]:
        # High accident count -> enforcement + education
        interventions.append(
            "Enforcement: Deploy speed cameras and drunk-driving checkpoints"
        )
        interventions.append(
            "Education: Run road safety campaigns targeting two-wheeler riders"
        )

    if row["flag_severity"]:
        # High severity -> engineering + emergency
        interventions.append(
            "Engineering: Add rumble strips, better signage, improve lighting"
        )
        interventions.append(
            "Emergency: Ensure trauma centres within 30 min of all black spots"
        )

    return " | ".join(interventions) if interventions else "Standard monitoring"


def plot_blackspot_map(bs_df: pd.DataFrame):
    """Horizontal bar chart showing risk level of top states."""
    top = bs_df.head(15).copy()
    color_map = {"Critical": "#d73027", "High": "#fc8d59",
                 "Medium": "#fee090", "Low": "#91bfdb"}
    colors = [color_map.get(r, "gray") for r in top["risk_level"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top["state"][::-1],
                   top["fatality_rate"][::-1],
                   color=colors[::-1], edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
    ax.set_xlabel("Fatality Rate (Deaths per 100 Accidents)")
    ax.set_title(f"Road Accident Black Spots by State ({YEAR_SHOW})",
                 fontweight="bold")

    from matplotlib.patches import Patch
    legend = [Patch(color=v, label=k) for k, v in color_map.items()]
    ax.legend(handles=legend, title="Risk Level",
              loc="lower right", fontsize=9)
    fig.tight_layout()
    path = PLOTS_DIR / "11_blackspot_risk_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot  ->  {path.name}")


def run_blackspot() -> pd.DataFrame:
    ensure_dirs()
    print("\n=== Black Spot Identification ===")

    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            "Cleaned CSV not found. Run python src/etl.py first."
        )

    df = pd.read_csv(CLEAN_CSV)
    bs = identify_blackspots(df, YEAR_SHOW)

    # Add intervention recommendations
    bs["recommended_interventions"] = bs.apply(recommend_interventions, axis=1)

    bs.to_csv(BLACKSPOTS_CSV, index=False)
    print(f"  Saved {len(bs)} rows  ->  {BLACKSPOTS_CSV}")

    n_critical = len(bs[bs["risk_level"] == "Critical"])
    n_high     = len(bs[bs["risk_level"] == "High"])
    n_bs       = len(bs[bs["is_blackspot"] == 1])

    print(f"\n  Results for {YEAR_SHOW}:")
    print(f"    Critical risk states : {n_critical}")
    print(f"    High risk states     : {n_high}")
    print(f"    Total black spots    : {n_bs}")

    print("\n  Top 5 black spot states:")
    cols = ["state", "total_accidents", "killed",
            "fatality_rate", "risk_level"]
    print(bs[bs["is_blackspot"] == 1].head(5)[cols].to_string(index=False))

    plot_blackspot_map(bs)
    print("\n=== Black spot analysis complete. ===")
    return bs


if __name__ == "__main__":
    bs = run_blackspot()
    print(f"\nFull results saved to: {BLACKSPOTS_CSV}")
