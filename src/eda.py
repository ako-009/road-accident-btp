"""
eda.py  (Exploratory Data Analysis)
-------------------------------------
Reads the cleaned CSV and produces 8 publication-quality plots
plus a summary CSV. All outputs go to outputs/plots/ and outputs/.

Run with:
    python src/eda.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (CLEAN_CSV, PLOTS_DIR, SUMMARY_CSV,
                        OUTPUT_DIR, ensure_dirs)

# ── Global plot style ─────────────────────────────────────────────
# Using a clean, professional style suitable for a BTP report
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE   = sns.color_palette("muted", 10)
FIG_DPI   = 150          # resolution for saved images
TOP_N     = 10           # how many states to show in bar charts
YEAR_SHOW = 2022         # which year to use for single-year charts


def load_data() -> pd.DataFrame:
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"Cleaned CSV not found at {CLEAN_CSV}. "
            "Run python src/etl.py first."
        )
    return pd.read_csv(CLEAN_CSV)


def save_fig(fig, name: str) -> Path:
    """Saves figure to outputs/plots/ and closes it to free memory."""
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot -> {path.name}")
    return path


# ── Plot 1: Top N states by total accidents ───────────────────────
def plot_top_accidents(df: pd.DataFrame):
    yr  = df[df["year"] == YEAR_SHOW]
    top = yr.nlargest(TOP_N, "total_accidents")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["state"][::-1], top["total_accidents"][::-1],
                   color=PALETTE[0], edgecolor="white")
    ax.bar_label(bars, fmt="%,.0f", padding=4, fontsize=9)
    ax.set_xlabel("Total Accidents")
    ax.set_title(f"Top {TOP_N} States by Total Accidents ({YEAR_SHOW})",
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return save_fig(fig, "01_top_accidents_by_state.png")


# ── Plot 2: Top N states by deaths ───────────────────────────────
def plot_top_deaths(df: pd.DataFrame):
    yr  = df[df["year"] == YEAR_SHOW]
    top = yr.nlargest(TOP_N, "killed")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["state"][::-1], top["killed"][::-1],
                   color=PALETTE[3], edgecolor="white")
    ax.bar_label(bars, fmt="%,.0f", padding=4, fontsize=9)
    ax.set_xlabel("Number of Deaths")
    ax.set_title(f"Top {TOP_N} States by Road Accident Deaths ({YEAR_SHOW})",
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return save_fig(fig, "02_top_deaths_by_state.png")


# ── Plot 3: Fatality rate by state ────────────────────────────────
def plot_fatality_rate(df: pd.DataFrame):
    yr  = df[df["year"] == YEAR_SHOW]
    top = yr.nlargest(TOP_N, "fatality_rate")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["state"][::-1], top["fatality_rate"][::-1],
                   color=PALETTE[1], edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_xlabel("Fatality Rate (Deaths per 100 Accidents)")
    ax.set_title(f"Top {TOP_N} States by Fatality Rate ({YEAR_SHOW})",
                 fontweight="bold")
    ax.axvline(yr["fatality_rate"].mean(), color="red",
               linestyle="--", linewidth=1.2, label="National average")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return save_fig(fig, "03_fatality_rate_by_state.png")


# ── Plot 4: Year-over-year national trend ─────────────────────────
def plot_yearly_trend(df: pd.DataFrame):
    yearly = (df.groupby("year")
               .agg(total_accidents=("total_accidents", "sum"),
                    killed=("killed", "sum"),
                    grievous=("grievous_injury", "sum"))
               .reset_index())

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(yearly["year"], yearly["total_accidents"],
             marker="o", color=PALETTE[0], linewidth=2.2,
             label="Total Accidents")
    ax2.plot(yearly["year"], yearly["killed"],
             marker="s", color=PALETTE[3], linewidth=2.2,
             linestyle="--", label="Deaths")

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total Accidents", color=PALETTE[0])
    ax2.set_ylabel("Deaths", color=PALETTE[3])
    ax1.set_title("National Road Accident Trend (2019–2023)",
                  fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    fig.tight_layout()
    return save_fig(fig, "04_yearly_national_trend.png")


# ── Plot 5: Injury severity breakdown (stacked bar) ───────────────
def plot_severity_breakdown(df: pd.DataFrame):
    yr = df[df["year"] == YEAR_SHOW]
    top = yr.nlargest(10, "total_accidents")[
        ["state", "fatal_accidents", "grievous_injury",
         "minor_hosp", "no_injury"]
    ].set_index("state")

    # Normalise to 100% so we compare proportions not raw counts
    top_pct = top.div(top.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    top_pct.plot(kind="barh", stacked=True, ax=ax,
                 color=[PALETTE[3], PALETTE[1], PALETTE[0], PALETTE[4]],
                 edgecolor="white")
    ax.set_xlabel("Percentage of Accidents (%)")
    ax.set_title(f"Accident Severity Breakdown — Top 10 States ({YEAR_SHOW})",
                 fontweight="bold")
    ax.legend(["Fatal", "Grievous", "Minor (Hosp.)", "No Injury"],
              loc="lower right", fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    fig.tight_layout()
    return save_fig(fig, "05_severity_breakdown.png")


# ── Plot 6: COVID impact — accidents before/during/after ──────────
def plot_covid_impact(df: pd.DataFrame):
    period = (df.groupby(["year", "period"])
               .agg(accidents=("total_accidents", "sum"),
                    deaths=("killed", "sum"))
               .reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"Pre-COVID": PALETTE[0],
              "COVID":     PALETTE[3],
              "Post-COVID": PALETTE[2]}

    for ax, col, label in zip(axes,
                               ["accidents", "deaths"],
                               ["Total Accidents", "Total Deaths"]):
        bar_colors = [colors.get(p, PALETTE[0])
                      for p in period["period"]]
        bars = ax.bar(period["year"].astype(str), period[col],
                      color=bar_colors, edgecolor="white")
        ax.bar_label(bars, fmt="%,.0f", padding=3, fontsize=8)
        ax.set_title(f"{label} by Year", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"))

    # Manual legend for COVID periods
    from matplotlib.patches import Patch
    legend_items = [Patch(color=v, label=k) for k, v in colors.items()]
    axes[1].legend(handles=legend_items, fontsize=9)

    fig.suptitle("Impact of COVID-19 on Road Accidents", fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "06_covid_impact.png")


# ── Plot 7: Night vs day accidents ───────────────────────────────
def plot_night_vs_day(df: pd.DataFrame):
    yr  = df[df["year"] == YEAR_SHOW]
    top = yr.nlargest(10, "total_accidents")[
        ["state", "total_accidents", "night_accidents"]
    ].copy()
    top["day_accidents"] = top["total_accidents"] - top["night_accidents"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x     = range(len(top))
    width = 0.4
    ax.barh([i + width/2 for i in x], top["day_accidents"],
            width, color=PALETTE[0], label="Day", edgecolor="white")
    ax.barh([i - width/2 for i in x], top["night_accidents"],
            width, color=PALETTE[5], label="Night", edgecolor="white")
    ax.set_yticks(list(x))
    ax.set_yticklabels(top["state"].tolist())
    ax.set_xlabel("Number of Accidents")
    ax.set_title(f"Day vs Night Accidents — Top 10 States ({YEAR_SHOW})",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return save_fig(fig, "07_day_vs_night.png")


# ── Plot 8: National highway vs state highway accidents ───────────
def plot_road_type(df: pd.DataFrame):
    yr  = df[df["year"] == YEAR_SHOW]
    top = yr.nlargest(10, "total_accidents")[
        ["state", "nh_accidents", "sh_accidents"]
    ].set_index("state")

    fig, ax = plt.subplots(figsize=(10, 6))
    top.plot(kind="barh", ax=ax,
             color=[PALETTE[0], PALETTE[2]], edgecolor="white")
    ax.set_xlabel("Number of Accidents")
    ax.set_title(f"National Highway vs State Highway Accidents ({YEAR_SHOW})",
                 fontweight="bold")
    ax.legend(["National Highway", "State Highway"], fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return save_fig(fig, "08_road_type_accidents.png")


# ── Summary CSV ───────────────────────────────────────────────────
def make_summary_csv(df: pd.DataFrame):
    """Saves a state-level summary CSV used by the AI agent later."""
    summary = (df.groupby("state")
                .agg(
                    total_accidents   = ("total_accidents",  "sum"),
                    total_killed      = ("killed",           "sum"),
                    total_grievous    = ("grievous_injury",  "sum"),
                    total_minor       = ("minor_hosp",       "sum"),
                    avg_fatality_rate = ("fatality_rate",    "mean"),
                    avg_severity      = ("severity_index",   "mean"),
                    years_recorded    = ("year",             "nunique"),
                )
                .round(2)
                .reset_index()
                .sort_values("total_killed", ascending=False))
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"  Saved summary CSV -> {SUMMARY_CSV.name}")
    return summary


# ── Master runner ─────────────────────────────────────────────────
def run_eda() -> dict:
    ensure_dirs()
    print("\n=== EDA Pipeline ===")
    df = load_data()
    print(f"  Loaded {len(df)} rows for analysis\n")

    plot_top_accidents(df)
    plot_top_deaths(df)
    plot_fatality_rate(df)
    plot_yearly_trend(df)
    plot_severity_breakdown(df)
    plot_covid_impact(df)
    plot_night_vs_day(df)
    plot_road_type(df)
    summary = make_summary_csv(df)

    plot_files = sorted([str(p) for p in PLOTS_DIR.glob("*.png")])
    print(f"\n=== EDA complete. {len(plot_files)} plots saved. ===")
    return {"status": "ok", "plots": plot_files, "summary": str(SUMMARY_CSV)}


if __name__ == "__main__":
    result = run_eda()
    print("\nPlots generated:")
    for p in result["plots"]:
        print(f"  {Path(p).name}")
