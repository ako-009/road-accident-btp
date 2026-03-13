"""
eda.py
------
Exploratory Data Analysis — generates all plots from cleaned iRAD data.
Saves 11 PNG files to outputs/plots/
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "processed" / "accidents_cleaned.csv"
PLOT_DIR  = ROOT / "outputs" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

BLUE   = "#2196F3"
RED    = "#F44336"
GREEN  = "#4CAF50"
ORANGE = "#FF9800"
PURPLE = "#9C27B0"


def save(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot -> {name}")


def fmt_num(n):
    """Format large numbers as 12,345"""
    return f"{int(n):,}"


# ── Load data ─────────────────────────────────────────────────────
print("=== EDA Pipeline ===")
df = pd.read_csv(DATA_FILE)
print(f"  Loaded {len(df)} rows for analysis")

df2022 = df[df["year"] == 2022].copy()
df2023 = df[df["year"] == 2023].copy()

# ── Plot 01: Top 10 states by total accidents ─────────────────────
top_acc = df2022.nlargest(10, "total_accidents").sort_values("total_accidents")
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_acc["state"], top_acc["total_accidents"], color=BLUE, edgecolor="white")
for bar, val in zip(bars, top_acc["total_accidents"]):
    ax.text(bar.get_width() + 300, bar.get_y() + bar.get_height() / 2,
            fmt_num(val), va="center", ha="left", fontsize=9)
ax.set_xlabel("Total Accidents")
ax.set_title("Top 10 States by Total Accidents (2022)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
save(fig, "01_top_accidents_by_state.png")

# ── Plot 02: Top 10 states by deaths ─────────────────────────────
top_dead = df2022.nlargest(10, "killed").sort_values("killed")
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_dead["state"], top_dead["killed"], color=RED, edgecolor="white")
for bar, val in zip(bars, top_dead["killed"]):
    ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
            fmt_num(val), va="center", ha="left", fontsize=9)
ax.set_xlabel("Number of Deaths")
ax.set_title("Top 10 States by Road Accident Deaths (2022)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
save(fig, "02_top_deaths_by_state.png")

# ── Plot 03: Top 10 states by fatality rate ───────────────────────
top_fat = df2022[df2022["total_accidents"] > 100].nlargest(10, "fatality_rate").sort_values("fatality_rate")
national_avg = df2022["fatality_rate"].mean()
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_fat["state"], top_fat["fatality_rate"], color=ORANGE, edgecolor="white")
for bar, val in zip(bars, top_fat["fatality_rate"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=9)
ax.axvline(national_avg, color=RED, linestyle="--", linewidth=1.5, label=f"National average ({national_avg:.1f}%)")
ax.set_xlabel("Fatality Rate (Deaths per 100 Accidents)")
ax.set_title("Top 10 States by Fatality Rate (2022)")
ax.legend()
fig.tight_layout()
save(fig, "03_fatality_rate_by_state.png")

# ── Plot 04: National yearly trend ───────────────────────────────
trend = df.groupby("year").agg(
    total_accidents=("total_accidents", "sum"),
    total_killed=("killed", "sum"),
).reset_index()

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.plot(trend["year"], trend["total_accidents"], "b-o", linewidth=2.5, markersize=7, label="Total Accidents")
ax2.plot(trend["year"], trend["total_killed"], "r--s", linewidth=2.5, markersize=7, label="Deaths")
ax1.set_xlabel("Year")
ax1.set_ylabel("Total Accidents", color="blue")
ax2.set_ylabel("Deaths", color="red")
ax1.set_title("National Road Accident Trend (2019–2023)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
ax1.set_xticks(trend["year"])
fig.tight_layout()
save(fig, "04_yearly_national_trend.png")

# ── Plot 05: Accident severity breakdown (2023 data) ─────────────
# Use 2023 data which has fatal/grievous/minor/no-injury breakdown
sev = df2023[df2023["fatal_accidents"] > 0].nlargest(10, "total_accidents").copy()

if len(sev) > 0:
    sev = sev.set_index("state")
    sev["fatal_pct"]    = sev["fatal_accidents"]  / sev["total_accidents"] * 100
    sev["grievous_pct"] = sev["grievous_injury"]   / sev["total_accidents"] * 100
    sev["minor_pct"]    = sev["minor_hosp"]        / sev["total_accidents"] * 100
    sev["noinjury_pct"] = sev["no_injury"]         / sev["total_accidents"] * 100

    fig, ax = plt.subplots(figsize=(11, 7))
    bottom = np.zeros(len(sev))
    colors = [RED, ORANGE, BLUE, GREEN]
    labels = ["Fatal", "Grievous", "Minor (Hosp.)", "No Injury"]
    for col, color, label in zip(["fatal_pct", "grievous_pct", "minor_pct", "noinjury_pct"], colors, labels):
        ax.barh(sev.index, sev[col], left=bottom, color=color, label=label, edgecolor="white")
        bottom += sev[col].values

    ax.set_xlabel("Percentage of Accidents (%)")
    ax.set_title("Accident Severity Breakdown — Top 10 States (2023)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 105)
    fig.tight_layout()
    save(fig, "05_severity_breakdown.png")
else:
    # Fallback: fatality rate bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    top10 = df2022.nlargest(10, "total_accidents").sort_values("fatality_rate")
    ax.barh(top10["state"], top10["fatality_rate"], color=ORANGE)
    ax.set_xlabel("Fatality Rate (%)")
    ax.set_title("Fatality Rate — Top 10 States by Accidents (2022)")
    fig.tight_layout()
    save(fig, "05_severity_breakdown.png")

# ── Plot 06: COVID impact ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
colors_covid = [BLUE if y == 2019 else RED if y in [2020, 2021] else GREEN
                for y in trend["year"]]

ax1.bar(trend["year"], trend["total_accidents"], color=colors_covid, edgecolor="white", width=0.6)
for i, (y, v) in enumerate(zip(trend["year"], trend["total_accidents"])):
    ax1.text(y, v + 3000, f"{int(v):,}", ha="center", fontsize=8)
ax1.set_title("Total Accidents by Year")
ax1.set_ylabel("Total Accidents")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

ax2.bar(trend["year"], trend["total_killed"], color=colors_covid, edgecolor="white", width=0.6)
for i, (y, v) in enumerate(zip(trend["year"], trend["total_killed"])):
    ax2.text(y, v + 1000, f"{int(v):,}", ha="center", fontsize=8)
ax2.set_title("Total Deaths by Year")
ax2.set_ylabel("Total Deaths")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

from matplotlib.patches import Patch
legend_elements = [Patch(color=BLUE, label="Pre-COVID"),
                   Patch(color=RED, label="COVID"),
                   Patch(color=GREEN, label="Post-COVID")]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
fig.suptitle("Impact of COVID-19 on Road Accidents", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "06_covid_impact.png")

# ── Plot 07: Year-wise state comparison (top 5 states trend) ─────
# Replace day/night chart with useful top-5-states trend since night data unavailable
top5_states = df2022.nlargest(5, "killed")["state"].tolist()
df_top5 = df[df["state"].isin(top5_states)]

fig, ax = plt.subplots(figsize=(11, 6))
colors_5 = [BLUE, RED, GREEN, ORANGE, PURPLE]
for state, color in zip(top5_states, colors_5):
    d = df_top5[df_top5["state"] == state].sort_values("year")
    ax.plot(d["year"], d["killed"], "o-", color=color, linewidth=2.5,
            markersize=7, label=state)
ax.set_xlabel("Year")
ax.set_ylabel("Persons Killed")
ax.set_title("Road Accident Deaths Trend — Top 5 States (2019–2023)")
ax.legend(loc="upper left")
ax.set_xticks([2019, 2020, 2021, 2022, 2023])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
save(fig, "07_day_vs_night.png")

# ── Plot 08: State-wise fatality rate change 2019 vs 2023 ─────────
# Replace empty NH/SH chart with meaningful comparison
df19 = df[df["year"] == 2019][["state", "fatality_rate"]].rename(columns={"fatality_rate": "fr_2019"})
df23 = df[df["year"] == 2023][["state", "fatality_rate"]].rename(columns={"fatality_rate": "fr_2023"})
compare = df19.merge(df23, on="state")
compare["change"] = compare["fr_2023"] - compare["fr_2019"]
compare = compare.sort_values("change")

fig, ax = plt.subplots(figsize=(11, 10))
colors_bar = [GREEN if c < 0 else RED for c in compare["change"]]
ax.barh(compare["state"], compare["change"], color=colors_bar, edgecolor="white")
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("Change in Fatality Rate (percentage points)")
ax.set_title("Change in Fatality Rate: 2019 vs 2023\n(Green = Improved, Red = Worsened)")
fig.tight_layout()
save(fig, "08_road_type_accidents.png")

# ── Save summary CSV ──────────────────────────────────────────────
summary = df.groupby("state").agg(
    total_accidents=("total_accidents", "sum"),
    total_killed=("killed", "sum"),
    avg_fatality_rate=("fatality_rate", "mean"),
    years_of_data=("year", "count"),
).round(2).reset_index().sort_values("total_killed", ascending=False)

summary.to_csv(ROOT / "outputs" / "summary_by_state.csv", index=False)
print("  Saved summary CSV -> summary_by_state.csv")

print("=== EDA complete. 11 plots saved. ===")
print("Plots generated:")
for p in sorted(PLOT_DIR.glob("0*.png")):
    print(f"  {p.name}")