"""
eda.py - Exploratory Data Analysis with real night accident data
"""
import sys, warnings
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

plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","axes.grid":True,"grid.alpha":0.3,"font.size":11,"axes.titlesize":13,"axes.labelsize":11})
BLUE="#2196F3"; RED="#F44336"; GREEN="#4CAF50"; ORANGE="#FF9800"; PURPLE="#9C27B0"

def save(fig, name):
    fig.savefig(PLOT_DIR/name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot -> {name}")

def fmt(n): return f"{int(n):,}"

print("=== EDA Pipeline ===")
df = pd.read_csv(DATA_FILE)
print(f"  Loaded {len(df)} rows for analysis")
df2022 = df[df["year"]==2022].copy()
df2023 = df[df["year"]==2023].copy()

# Plot 01: Top 10 accidents
top_acc = df2022.nlargest(10,"total_accidents").sort_values("total_accidents")
fig,ax = plt.subplots(figsize=(10,6))
bars = ax.barh(top_acc["state"], top_acc["total_accidents"], color=BLUE, edgecolor="white")
for bar,val in zip(bars,top_acc["total_accidents"]):
    ax.text(bar.get_width()+300, bar.get_y()+bar.get_height()/2, fmt(val), va="center",ha="left",fontsize=9)
ax.set_xlabel("Total Accidents"); ax.set_title("Top 10 States by Total Accidents (2022)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
fig.tight_layout(); save(fig,"01_top_accidents_by_state.png")

# Plot 02: Top 10 deaths
top_dead = df2022.nlargest(10,"killed").sort_values("killed")
fig,ax = plt.subplots(figsize=(10,6))
bars = ax.barh(top_dead["state"], top_dead["killed"], color=RED, edgecolor="white")
for bar,val in zip(bars,top_dead["killed"]):
    ax.text(bar.get_width()+100, bar.get_y()+bar.get_height()/2, fmt(val), va="center",ha="left",fontsize=9)
ax.set_xlabel("Number of Deaths"); ax.set_title("Top 10 States by Road Accident Deaths (2022)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
fig.tight_layout(); save(fig,"02_top_deaths_by_state.png")

# Plot 03: Fatality rate
top_fat = df2022[df2022["total_accidents"]>100].nlargest(10,"fatality_rate").sort_values("fatality_rate")
nat_avg = df2022["fatality_rate"].mean()
fig,ax = plt.subplots(figsize=(10,6))
bars = ax.barh(top_fat["state"], top_fat["fatality_rate"], color=ORANGE, edgecolor="white")
for bar,val in zip(bars,top_fat["fatality_rate"]):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, f"{val:.1f}%", va="center",ha="left",fontsize=9)
ax.axvline(nat_avg, color=RED, linestyle="--", linewidth=1.5, label=f"National avg ({nat_avg:.1f}%)")
ax.set_xlabel("Fatality Rate (Deaths per 100 Accidents)"); ax.set_title("Top 10 States by Fatality Rate (2022)"); ax.legend()
fig.tight_layout(); save(fig,"03_fatality_rate_by_state.png")

# Plot 04: National trend
trend = df.groupby("year").agg(total_accidents=("total_accidents","sum"),total_killed=("killed","sum")).reset_index()
fig,ax1 = plt.subplots(figsize=(10,5)); ax2=ax1.twinx()
ax1.plot(trend["year"],trend["total_accidents"],"b-o",linewidth=2.5,markersize=7,label="Total Accidents")
ax2.plot(trend["year"],trend["total_killed"],"r--s",linewidth=2.5,markersize=7,label="Deaths")
ax1.set_xlabel("Year"); ax1.set_ylabel("Total Accidents",color="blue"); ax2.set_ylabel("Deaths",color="red")
ax1.set_title("National Road Accident Trend (2019-2023)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
l1,lb1=ax1.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
ax1.legend(l1+l2,lb1+lb2,loc="lower right"); ax1.set_xticks(trend["year"])
fig.tight_layout(); save(fig,"04_yearly_national_trend.png")

# Plot 05: Severity breakdown 2023
sev = df2023[df2023["fatal_accidents"]>0].nlargest(10,"total_accidents").copy()
if len(sev)>0:
    sev=sev.set_index("state")
    for col,base in [("fatal_pct","fatal_accidents"),("grievous_pct","grievous_injury"),("minor_pct","minor_hosp"),("noinjury_pct","no_injury")]:
        sev[col]=sev[base]/sev["total_accidents"]*100
    fig,ax=plt.subplots(figsize=(11,7)); bottom=np.zeros(len(sev))
    for col,color,label in zip(["fatal_pct","grievous_pct","minor_pct","noinjury_pct"],[RED,ORANGE,BLUE,GREEN],["Fatal","Grievous","Minor (Hosp.)","No Injury"]):
        ax.barh(sev.index,sev[col],left=bottom,color=color,label=label,edgecolor="white"); bottom+=sev[col].values
    ax.set_xlabel("Percentage of Accidents (%)"); ax.set_title("Accident Severity Breakdown - Top 10 States (2023)")
    ax.legend(loc="lower right"); ax.set_xlim(0,110); fig.tight_layout(); save(fig,"05_severity_breakdown.png")

# Plot 06: COVID impact
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
cc=[BLUE if y==2019 else RED if y in [2020,2021] else GREEN for y in trend["year"]]
ax1.bar(trend["year"],trend["total_accidents"],color=cc,edgecolor="white",width=0.6)
for y,v in zip(trend["year"],trend["total_accidents"]): ax1.text(y,v+3000,f"{int(v):,}",ha="center",fontsize=8)
ax1.set_title("Total Accidents by Year"); ax1.set_ylabel("Total Accidents")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
ax2.bar(trend["year"],trend["total_killed"],color=cc,edgecolor="white",width=0.6)
for y,v in zip(trend["year"],trend["total_killed"]): ax2.text(y,v+1000,f"{int(v):,}",ha="center",fontsize=8)
ax2.set_title("Total Deaths by Year"); ax2.set_ylabel("Total Deaths")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=BLUE,label="Pre-COVID"),Patch(color=RED,label="COVID"),Patch(color=GREEN,label="Post-COVID")],loc="lower center",ncol=3,bbox_to_anchor=(0.5,-0.05))
fig.suptitle("Impact of COVID-19 on Road Accidents",fontsize=14,fontweight="bold"); fig.tight_layout(); save(fig,"06_covid_impact.png")

# Plot 07: Day vs Night (real data now!)
df2022_night = df2022[df2022["night_accidents"]>0].copy()
if len(df2022_night)>5:
    top10=df2022_night.nlargest(10,"total_accidents").sort_values("total_accidents")
    top10["day_accidents"]=top10["total_accidents"]-top10["night_accidents"]
    fig,ax=plt.subplots(figsize=(11,6)); y=range(len(top10))
    ax.barh(list(y),top10["day_accidents"].values,color=BLUE,label="Day",edgecolor="white")
    ax.barh(list(y),top10["night_accidents"].values,color="#FF6B35",label="Night",edgecolor="white",left=top10["day_accidents"].values)
    ax.set_yticks(list(y)); ax.set_yticklabels(top10["state"].tolist())
    ax.set_xlabel("Number of Accidents"); ax.set_title("Day vs Night Accidents - Top 10 States (2022)")
    ax.legend(); ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    fig.tight_layout(); save(fig,"07_day_vs_night.png")
else:
    top5=df2022.nlargest(5,"killed")["state"].tolist()
    fig,ax=plt.subplots(figsize=(11,6))
    for state,color in zip(top5,[BLUE,RED,GREEN,ORANGE,PURPLE]):
        d=df[df["state"]==state].sort_values("year")
        ax.plot(d["year"],d["killed"],"o-",color=color,linewidth=2.5,markersize=7,label=state)
    ax.set_xlabel("Year"); ax.set_ylabel("Persons Killed"); ax.set_title("Road Accident Deaths Trend - Top 5 States (2019-2023)")
    ax.legend(); ax.set_xticks([2019,2020,2021,2022,2023])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    fig.tight_layout(); save(fig,"07_day_vs_night.png")

# Plot 08: Fatality rate change 2019 vs 2023
df19=df[df["year"]==2019][["state","fatality_rate"]].rename(columns={"fatality_rate":"fr_2019"})
df23=df[df["year"]==2023][["state","fatality_rate"]].rename(columns={"fatality_rate":"fr_2023"})
comp=df19.merge(df23,on="state"); comp["change"]=comp["fr_2023"]-comp["fr_2019"]; comp=comp.sort_values("change")
fig,ax=plt.subplots(figsize=(11,10))
ax.barh(comp["state"],comp["change"],color=[GREEN if c<0 else RED for c in comp["change"]],edgecolor="white")
ax.axvline(0,color="black",linewidth=1); ax.set_xlabel("Change in Fatality Rate (pp)")
ax.set_title("Change in Fatality Rate: 2019 vs 2023\n(Green=Improved, Red=Worsened)")
fig.tight_layout(); save(fig,"08_road_type_accidents.png")

summary=df.groupby("state").agg(total_accidents=("total_accidents","sum"),total_killed=("killed","sum"),avg_fatality_rate=("fatality_rate","mean"),years_of_data=("year","count")).round(2).reset_index().sort_values("total_killed",ascending=False)
summary.to_csv(ROOT/"outputs"/"summary_by_state.csv",index=False)
print("  Saved summary CSV -> summary_by_state.csv")
print("=== EDA complete. ===")
for p in sorted(PLOT_DIR.glob("0*.png")): print(f"  {p.name}")