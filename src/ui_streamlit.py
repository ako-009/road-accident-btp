"""
ui_streamlit.py - 4-tab Streamlit dashboard for Road Accident BTP project.
Run with: streamlit run src/ui_streamlit.py
"""

import sys
import json
import requests
import pandas as pd
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from src.config import PLOTS_DIR, OUTPUT_DIR

st.set_page_config(
    page_title="India Road Safety Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"


def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def api_post(endpoint, payload):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## India Road Safety Dashboard")
    st.markdown("**India Road Safety Analysis**")
    st.markdown("---")
    health = api_get("/health")
    if health.get("status") == "ok":
        st.success("API connected")
    else:
        st.error("API offline. Run:\nuvicorn src.agent_api:app --port 8000")
    st.markdown("---")
    st.markdown("**Data source:** iRAD / MoRTH")
    st.markdown("**Years covered:** 2019 – 2023")
    st.markdown("**Model:** Poisson Regression")
    st.markdown("**Framework:** 4E (Campbell Review)")
    st.markdown("---")
    st.markdown("**Project:** BTP Road Safety")
    st.markdown("**Institute:** IIT Kharagpur")
    st.markdown("---")
    st.caption("Data sourced from official MoRTH Road Accidents in India publications.")


tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "EDA Plots", "Analysis", "AI Chat"
])


# ── TAB 1: OVERVIEW ───────────────────────────────────────────────
with tab1:
    st.markdown("## National Road Accident Overview")
    st.markdown(
        "Data from the **Integrated Road Accident Database (iRAD)** "
        "covering all 36 states and union territories of India. "
        "Source: MoRTH Road Accidents in India 2023."
    )

    totals   = api_get("/totals")
    totals22 = api_get("/totals") # we'll use trend for YoY
    trend_d  = api_get("/trend")

    if totals.get("status") == "ok":
        year = totals.get("latest_year", "N/A")

        # ── YoY change calculation ────────────────────────────────
        yoy_acc  = None
        yoy_kill = None
        if trend_d.get("status") == "ok":
            tr = trend_d["trend"]
            if len(tr) >= 2:
                prev = tr[-2]
                curr = tr[-1]
                if prev["total_accidents"] > 0:
                    yoy_acc  = round((curr["total_accidents"] - prev["total_accidents"]) / prev["total_accidents"] * 100, 1)
                if prev["total_killed"] > 0:
                    yoy_kill = round((curr["total_killed"] - prev["total_killed"]) / prev["total_killed"] * 100, 1)

        st.markdown(f"#### Statistics for {year}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Total Accidents",
            f"{totals['total_accidents']:,}",
            delta=f"{yoy_acc:+.1f}% vs 2022" if yoy_acc is not None else None,
            delta_color="inverse",
            help="Total road accidents reported across all 36 states/UTs"
        )
        c2.metric(
            "Total Deaths",
            f"{totals['total_killed']:,}",
            delta=f"{yoy_kill:+.1f}% vs 2022" if yoy_kill is not None else None,
            delta_color="inverse",
            help="Total persons killed in road accidents"
        )
        c3.metric(
            "Grievous Injuries",
            f"{totals['total_grievous']:,}",
            help="Persons with grievous injuries (2023 data)"
        )
        c4.metric(
            "Avg Fatality Rate",
            f"{totals['avg_fatality_rate']:.1f}%",
            help="Deaths per 100 accidents — national average across all states"
        )

        c5, c6, c7, c8 = st.columns(4)
        c5.metric(
            "States Covered",
            36,
            help="All 28 states + 8 union territories of India"
        )
        c6.metric(
            "Minor Injuries",
            f"{totals['total_minor']:,}",
            help="Persons with minor injuries in road accidents (2023)"
        )
        c7.metric(
            "Years of Data",
            len(totals.get("years_in_data", [])),
            help="Data available from 2019 to 2023"
        )
        c8.metric(
            "Data Period",
            f"2019 – {year}"
        )

        st.markdown("---")

        # ── Key Insights Box ──────────────────────────────────────
        st.markdown("#### Key Insights")

        # Get blackspot data for insights
        bs_data  = api_get("/blackspots", params={"risk_level": "Critical"})
        top_acc  = api_get("/top-states", params={"by": "total_accidents", "year": 2023, "n": 1})
        top_fat  = api_get("/top-states", params={"by": "fatality_rate", "year": 2022, "n": 1})
        top_kill = api_get("/top-states", params={"by": "killed", "year": 2023, "n": 1})

        col_i1, col_i2, col_i3, col_i4 = st.columns(4)

        with col_i1:
            if top_kill.get("status") == "ok" and top_kill.get("top_states"):
                s = top_kill["top_states"][0]
                st.info(f"**Most Deaths (2023)**\n\n{s['state']}\n\n{s['killed']:,} persons killed")

        with col_i2:
            if top_acc.get("status") == "ok" and top_acc.get("top_states"):
                s = top_acc["top_states"][0]
                st.info(f"**Most Accidents (2023)**\n\n{s['state']}\n\n{s['total_accidents']:,} accidents")

        with col_i3:
            if top_fat.get("status") == "ok" and top_fat.get("top_states"):
                s = top_fat["top_states"][0]
                st.warning(f"**Highest Fatality Rate**\n\n{s['state']}\n\n{s['fatality_rate']}% fatality rate")

        with col_i4:
            st.warning(
                f"**Night Accidents (2023)**\n\n"
                f"~42% of all accidents\n\n"
                f"occur between 18:00–06:00 hrs"
            )

        st.markdown("---")

        # ── Top 10 States Table ───────────────────────────────────
        st.markdown("### Top 10 States by Deaths (2023)")
        st.caption("Ranked by total persons killed. Severity Score = killed / total_accidents (higher = more severe).")

        top = api_get("/top-states", params={"by": "killed", "year": 2023, "n": 10})
        if top.get("status") == "ok":
            df_top = pd.DataFrame(top["top_states"])
            df_top.columns = [
                "State", "Total Accidents", "Deaths",
                "Fatality Rate (%)", "Severity Score"
            ]

            # Add risk level column
            def risk_label(fat_rate):
                if fat_rate >= 50: return "🔴 Critical"
                elif fat_rate >= 35: return "🟠 High"
                elif fat_rate >= 20: return "🟡 Medium"
                else: return "🟢 Low"

            df_top["Risk Level"] = df_top["Fatality Rate (%)"].apply(risk_label)
            df_top.index = range(1, len(df_top) + 1)
            st.dataframe(df_top, use_container_width=True)

        st.markdown("---")

        # ── COVID Recovery Note ───────────────────────────────────
        if trend_d.get("status") == "ok":
            tr = trend_d["trend"]
            min_yr = min(tr, key=lambda x: x["total_accidents"])
            max_yr = max(tr, key=lambda x: x["total_accidents"])
            recovery_pct = round(
                (max_yr["total_accidents"] - min_yr["total_accidents"]) /
                min_yr["total_accidents"] * 100, 1
            )
            st.info(
                f"**COVID-19 Impact & Recovery:** Accidents dropped to a low of "
                f"{min_yr['total_accidents']:,} in {int(min_yr['year'])} due to lockdowns, "
                f"then recovered by {recovery_pct}% to {max_yr['total_accidents']:,} "
                f"in {int(max_yr['year'])}. Road safety remains a critical public health priority."
            )

        st.markdown("---")

    else:
        st.warning("Could not load totals. Make sure the API is running.")

    # ── About iRAD ────────────────────────────────────────────────
    st.markdown("### About iRAD and the 4E Framework")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
The **iRAD (Integrated Road Accident Database)** was launched on
Jan 13, 2020 by MoRTH. Developed by NIC, it covers all 36 states and UTs.

**Key features:**
- Mobile-based on-site accident data collection
- Black spot identification using GIS grid analysis
- eDAR for cashless victim treatment
- Integration with CCTNS, Vahan, Sarathi, NHA, eCourts
- Real-time accident reporting by traffic police
        """)
    with col_b:
        st.markdown("""
**The 4E Framework (Campbell Systematic Review):**

| E | Description | Effectiveness |
|---|---|---|
| Engineering | Road design, barriers, signage, lighting | 20–40% reduction |
| Enforcement | Speed cameras, drunk-driving checks | 10–20% reduction |
| Education | Campaigns, driver training | 5–15% reduction |
| Emergency | Ambulance response, trauma centres | 15–20% reduction |

Evidence shows combining all 4Es reduces fatalities by **30–50%**.
        """)


# ── TAB 2: EDA PLOTS ──────────────────────────────────────────────
with tab2:
    st.markdown("## Exploratory Data Analysis")
    st.markdown(
        "All charts generated from the cleaned iRAD dataset (MoRTH 2019–2023). "
        "Real data for all 36 states."
    )

    plot_data  = api_get("/plots")
    plot_files = plot_data.get("plots", [])
    if not plot_files:
        plot_files = sorted([p.name for p in PLOTS_DIR.glob("*.png")])

    if not plot_files:
        st.warning("No plots found. Run python src/eda.py first.")
    else:
        descriptions = {
            "01_top_accidents_by_state.png": "Top 10 states by total accidents (2022). Tamil Nadu leads with 64,105 accidents.",
            "02_top_deaths_by_state.png":    "Top 10 states by road accident deaths (2022). Uttar Pradesh leads with 22,595 deaths.",
            "03_fatality_rate_by_state.png": "States with highest fatality rate (deaths per 100 accidents). Red dashed line = national average.",
            "04_yearly_national_trend.png":  "National trend 2019–2023. COVID-19 caused a sharp drop in 2020. Accidents and deaths have risen since.",
            "05_severity_breakdown.png":     "Accident severity breakdown for top 10 states (2023). Shows fatal, grievous, minor and no-injury accidents.",
            "06_covid_impact.png":           "COVID-19 impact on road accidents. 2020 lockdowns reduced accidents by 18.5% nationally.",
            "07_day_vs_night.png":           "Day vs Night accidents — Top 10 states (2022). Night = 18:00 to 06:00 hrs. ~42% of accidents are at night.",
            "08_road_type_accidents.png":    "Change in fatality rate: 2019 vs 2023. Green = improved, Red = worsened compared to 2019.",
            "09_actual_vs_predicted.png":    "Poisson regression model: actual vs predicted accident counts. Points near diagonal = better fit.",
            "10_feature_importance.png":     "Poisson model feature coefficients. Green = increases accidents, Red = decreases accident count.",
            "11_blackspot_risk_map.png":     "Road accident black spot map. Critical and High risk states need immediate intervention.",
        }
        pairs = [plot_files[i:i+2] for i in range(0, len(plot_files), 2)]
        for pair in pairs:
            cols = st.columns(2)
            for col, fname in zip(cols, pair):
                fpath = PLOTS_DIR / fname
                if fpath.exists():
                    with col:
                        st.image(str(fpath), use_container_width=True)
                        st.caption(descriptions.get(fname, ""))
                        st.markdown("---")


# ── TAB 3: ANALYSIS ───────────────────────────────────────────────
with tab3:
    st.markdown("## Deep Analysis")
    sec1, sec2, sec3 = st.tabs(["Black Spots", "Yearly Trend", "Model Metrics"])

    with sec1:
        st.markdown("### Road Accident Black Spots")
        st.markdown(
            "States in top 25% nationally on 2 or more risk criteria are classified as black spots. "
            "Criteria include: total accidents, deaths, fatality rate, and severity index."
        )

        risk_filter = st.selectbox(
            "Filter by risk level",
            ["all", "Critical", "High", "Medium", "Low"]
        )
        bs_data = api_get("/blackspots", params={"risk_level": risk_filter})

        if bs_data.get("status") == "ok" and bs_data.get("blackspots"):
            df_bs = pd.DataFrame(bs_data["blackspots"])

            # Add emoji risk labels
            def risk_emoji(level):
                return {"Critical": "🔴 Critical", "High": "🟠 High",
                        "Medium": "🟡 Medium", "Low": "🟢 Low"}.get(level, level)

            df_display = df_bs[["state","total_accidents","killed","fatality_rate","risk_level"]].copy()
            df_display["risk_level"] = df_display["risk_level"].apply(risk_emoji)
            df_display = df_display.rename(columns={
                "state":           "State",
                "total_accidents": "Accidents (2022)",
                "killed":          "Deaths (2022)",
                "fatality_rate":   "Fatality Rate (%)",
                "risk_level":      "Risk Level",
            })
            st.dataframe(df_display, use_container_width=True)

            # Summary counts
            counts = df_bs["risk_level"].value_counts()
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Critical States", counts.get("Critical", 0))
            sc2.metric("High Risk States", counts.get("High", 0))
            sc3.metric("Medium Risk States", counts.get("Medium", 0))
            sc4.metric("Total Black Spots", len(df_bs))

            st.markdown("#### Recommended Interventions")
            st.caption("Select a state to see evidence-based interventions from the Campbell Systematic Review.")
            selected = st.selectbox("Select a state", df_bs["state"].tolist())
            if selected:
                row = df_bs[df_bs["state"] == selected].iloc[0]
                st.markdown(f"**Interventions for {selected}** (Fatality Rate: {row['fatality_rate']}%)")
                for item in row.get("recommended_interventions", "").split(" | "):
                    if item.strip():
                        parts = item.split(":", 1)
                        if len(parts) == 2:
                            cat, detail = parts
                            color_map = {
                                "Engineering": "🔧",
                                "Enforcement": "🚔",
                                "Education":   "📚",
                                "Emergency":   "🚑",
                            }
                            icon = color_map.get(cat.strip(), "•")
                            st.markdown(f"{icon} **{cat.strip()}:** {detail.strip()}")
        else:
            st.info("No black spots found for the selected filter.")

    with sec2:
        st.markdown("### National Year-over-Year Trend")
        st.caption("Road accident statistics from 2019 to 2023. Source: MoRTH Annual Reports.")
        trend_data = api_get("/trend")
        if trend_data.get("status") == "ok":
            df_trend = pd.DataFrame(trend_data["trend"])
            df_trend.columns = [
                "Year", "Total Accidents", "Total Deaths", "Avg Fatality Rate (%)"
            ]

            # Add YoY change columns
            df_trend["Accident Change (%)"] = df_trend["Total Accidents"].pct_change().mul(100).round(1)
            df_trend["Death Change (%)"]    = df_trend["Total Deaths"].pct_change().mul(100).round(1)
            df_trend = df_trend.set_index("Year")

            st.dataframe(df_trend, use_container_width=True)
            st.line_chart(df_trend[["Total Accidents", "Total Deaths"]])
            st.info(
                "**2020:** Drop due to COVID-19 lockdowns — traffic reduced by ~60%. "
                "**2021-2023:** Gradual recovery as mobility normalised. "
                "**2023:** Accidents at highest level since 2019."
            )
        else:
            st.warning("Could not load trend data.")

    with sec3:
        st.markdown("### Poisson Regression Model Performance")
        st.markdown(
            "Poisson regression is used to predict accident counts because accidents are "
            "whole non-negative numbers (count data). It is the standard model used in "
            "official road safety research worldwide including MoRTH and WHO studies."
        )
        metrics_data = api_get("/metrics")
        if metrics_data.get("status") == "ok":
            m = metrics_data["metrics"]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(
                "MAE",
                f"{float(m['mae']):,.0f} accidents",
                help="Mean Absolute Error — average prediction error per state per year"
            )
            mc2.metric(
                "RMSE",
                f"{float(m['rmse']):,.0f} accidents",
                help="Root Mean Square Error — penalises large errors more heavily"
            )
            mc3.metric(
                "MAPE",
                f"{float(m['mape_pct']):.1f}%",
                help="Mean Absolute Percentage Error — relative prediction error"
            )

            st.markdown("**Features used in the model:**")
            feat_descriptions = {
                "is_covid_year":  "Whether the year is 2020 or 2021 (COVID period)",
                "is_large_state": "Whether the state has population > 50 million",
                "nh_share":       "Proportion of accidents on National Highways",
                "night_share":    "Proportion of accidents occurring at night (18:00–06:00)",
                "fatality_rate":  "Deaths per 100 accidents",
                "severity_index": "Composite severity score (killed / total accidents)",
            }
            for feat in str(m.get("features_used", "")).split(", "):
                feat = feat.strip()
                desc = feat_descriptions.get(feat, "")
                if desc:
                    st.markdown(f"- `{feat}` — {desc}")
                else:
                    st.markdown(f"- `{feat}`")

            st.caption(
                "Note: High MAPE is expected with state-level count data due to extreme "
                "variance between large states (Tamil Nadu: 67,213) and small UTs (Lakshadweep: 3). "
                "The model captures broad patterns including COVID impact and state size effects."
            )
        else:
            st.warning("Run python src/train_poisson.py first.")


# ── TAB 4: AI CHAT ────────────────────────────────────────────────
with tab4:
    st.markdown("## AI Road Safety Assistant")
    st.markdown(
        "Ask anything about road accidents in India. "
        "The assistant searches the iRAD knowledge base and answers based on real MoRTH data."
    )

    prompts_path = Path(__file__).parent / "canned_prompts.json"
    canned = []
    if prompts_path.exists():
        with open(prompts_path, encoding="utf-8") as f:
            canned = json.load(f)

    preset_question = ""
    if canned:
        st.markdown("#### Quick questions")
        labels = ["-- choose a preset --"] + [p["label"] for p in canned]
        chosen = st.selectbox("", labels, label_visibility="collapsed")
        if chosen != "-- choose a preset --":
            match = next((p for p in canned if p["label"] == chosen), None)
            if match:
                preset_question = match["question"]

    st.markdown("#### Or type your own question")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_area(
        "",
        value=preset_question,
        height=80,
        placeholder="e.g. Which states have the highest fatality rate in 2022?",
        label_visibility="collapsed",
    )

    col_ask, col_clear = st.columns([1, 5])
    with col_ask:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    if ask_clicked and user_input.strip():
        with st.spinner("Searching knowledge base and generating answer..."):
            response = api_post("/chat", {"question": user_input.strip()})
        if response.get("status") == "error":
            st.error(f"Error: {response.get('message', 'Unknown')}")
        else:
            st.session_state.chat_history.append({
                "question": response.get("question", user_input),
                "answer":   response.get("answer", "No answer returned."),
                "sources":  response.get("sources", []),
            })

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Conversation")
        for entry in reversed(st.session_state.chat_history):
            st.markdown(
                f'<div style="background:#1a3a4a;padding:10px;border-radius:8px;'
                f'margin-bottom:6px;color:#ffffff"><b>You:</b> {entry["question"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#1a3d1a;padding:10px;border-radius:8px;'
                f'margin-bottom:4px;color:#ffffff"><b>Assistant:</b> {entry["answer"]}</div>',
                unsafe_allow_html=True,
            )
            if entry.get("sources"):
                relevant = [s for s in entry["sources"] if s["score"] > 0.55]
                with st.expander(f"Sources ({len(relevant)} relevant documents)"):
                    for j, src in enumerate(relevant, 1):
                        st.markdown(
                            f"**[{j}] {src['id']}** "
                            f"*(source: {src['source']}, score: {src['score']:.3f})*"
                        )
                        st.markdown(
                            f"<small>{src['text'][:250]}...</small>",
                            unsafe_allow_html=True,
                        )
            st.markdown("---")
    elif not ask_clicked:
        st.markdown("""
#### How to use this assistant

1. Pick a quick question from the dropdown, or type your own
2. Click **Ask** to get an answer

The assistant searches iRAD data, black spot results,
Campbell Review interventions, and Poisson model predictions.

**Example questions:**
- Which state has the highest fatality rate?
- Tell me about road accidents in Uttar Pradesh
- What engineering interventions reduce road deaths?
- How did COVID-19 affect road accidents in India?
- Which states are classified as black spots?
- Explain the Poisson regression model used
        """)