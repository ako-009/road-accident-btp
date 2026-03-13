"""
tools.py
---------
Tool functions that the AI agent can call to get live data.

Each function:
  - Does one specific task (run EDA, get totals, get blackspots, etc.)
  - Returns a clean dictionary with results
  - Handles errors gracefully so the API never crashes

The agent_api.py imports and calls these functions based on
what the user is asking about.
"""

import sys
import traceback
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    CLEAN_CSV, SUMMARY_CSV, BLACKSPOTS_CSV,
    PREDICTIONS_CSV, OUTPUT_DIR, ensure_dirs,
)


def get_national_totals() -> dict:
    """
    Returns headline national statistics from the cleaned data.
    Used to answer questions like 'How many accidents in India?'
    """
    try:
        df = pd.read_csv(CLEAN_CSV)
        latest_year = int(df["year"].max())
        latest      = df[df["year"] == latest_year]

        return {
            "status":           "ok",
            "latest_year":      latest_year,
            "total_accidents":  int(latest["total_accidents"].sum()),
            "total_killed":     int(latest["killed"].sum()),
            "total_grievous":   int(latest["grievous_injury"].sum()),
            "total_minor":      int(latest["minor_hosp"].sum()),
            "avg_fatality_rate": round(float(latest["fatality_rate"].mean()), 2),
            "states_covered":   int(latest["state"].nunique()),
            "years_in_data":    sorted(df["year"].unique().tolist()),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_top_states(by: str = "killed", year: int = 2022, n: int = 10) -> dict:
    """
    Returns the top N states ranked by the chosen metric.
    by options: 'killed', 'total_accidents', 'fatality_rate', 'severity_index'
    """
    try:
        df = pd.read_csv(CLEAN_CSV)
        yr = df[df["year"] == year]
        if yr.empty:
            return {"status": "error", "message": f"No data for year {year}"}

        valid_cols = ["killed", "total_accidents",
                      "fatality_rate", "severity_index"]
        if by not in valid_cols:
            by = "killed"

        top = (yr.nlargest(n, by)
                 [["state", "total_accidents", "killed",
                   "fatality_rate", "severity_index"]]
                 .round(2)
                 .to_dict(orient="records"))

        return {
            "status": "ok",
            "year":   year,
            "ranked_by": by,
            "top_states": top,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_state_summary(state_name: str) -> dict:
    """
    Returns a detailed summary for one specific state.
    Used to answer questions like 'Tell me about road accidents in Delhi'
    """
    try:
        df = pd.read_csv(CLEAN_CSV)

        # Case-insensitive state name matching
        mask = df["state"].str.lower() == state_name.lower()
        state_df = df[mask]

        if state_df.empty:
            # Try partial match
            mask = df["state"].str.lower().str.contains(
                state_name.lower(), na=False
            )
            state_df = df[mask]

        if state_df.empty:
            return {
                "status":  "error",
                "message": f"State '{state_name}' not found in data.",
            }

        actual_name = state_df["state"].iloc[0]

        yearly = (state_df.groupby("year")
                          .agg(accidents=("total_accidents", "sum"),
                               killed=("killed", "sum"),
                               fatality_rate=("fatality_rate", "mean"))
                          .round(2)
                          .reset_index()
                          .to_dict(orient="records"))

        return {
            "status":           "ok",
            "state":            actual_name,
            "total_accidents":  int(state_df["total_accidents"].sum()),
            "total_killed":     int(state_df["killed"].sum()),
            "avg_fatality_rate": round(float(state_df["fatality_rate"].mean()), 2),
            "worst_year":       int(state_df.loc[
                                    state_df["total_accidents"].idxmax(), "year"
                                ]),
            "yearly_breakdown": yearly,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_blackspots(risk_level: str = "all") -> dict:
    """
    Returns black spot states.
    risk_level options: 'Critical', 'High', 'Medium', 'Low', 'all'
    """
    try:
        if not BLACKSPOTS_CSV.exists():
            return {
                "status":  "error",
                "message": "Blackspot data not found. Run python src/blackspot.py first.",
            }

        bs = pd.read_csv(BLACKSPOTS_CSV)

        if risk_level != "all":
            bs = bs[bs["risk_level"].str.lower() == risk_level.lower()]

        result = (bs[bs["is_blackspot"] == 1]
                    [["state", "total_accidents", "killed",
                      "fatality_rate", "risk_level",
                      "recommended_interventions"]]
                    .to_dict(orient="records"))

        return {
            "status":      "ok",
            "risk_filter": risk_level,
            "count":       len(result),
            "blackspots":  result,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def run_eda_tool() -> dict:
    """
    Runs the EDA pipeline and returns the list of generated plot filenames.
    Useful when the user asks to regenerate plots.
    """
    try:
        from src.eda import run_eda
        result = run_eda()
        return result
    except Exception as e:
        return {"status": "error", "message": traceback.format_exc()}


def get_plot_list() -> dict:
    """Returns the list of all available plot files."""
    try:
        from src.config import PLOTS_DIR
        plots = sorted([p.name for p in PLOTS_DIR.glob("*.png")])
        return {
            "status": "ok",
            "count":  len(plots),
            "plots":  plots,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_model_metrics() -> dict:
    """Returns the saved Poisson model performance metrics."""
    try:
        metrics_path = OUTPUT_DIR / "model_metrics.csv"
        if not metrics_path.exists():
            return {
                "status":  "error",
                "message": "Model metrics not found. Run python src/train_poisson.py first.",
            }
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        return {"status": "ok", "metrics": metrics}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_yearly_trend() -> dict:
    """Returns national year-over-year accident and death counts."""
    try:
        df = pd.read_csv(CLEAN_CSV)
        trend = (df.groupby("year")
                   .agg(total_accidents=("total_accidents", "sum"),
                        total_killed=("killed", "sum"),
                        avg_fatality_rate=("fatality_rate", "mean"))
                   .round(2)
                   .reset_index()
                   .to_dict(orient="records"))
        return {"status": "ok", "trend": trend}
    except Exception as e:
        return {"status": "error", "message": str(e)}
