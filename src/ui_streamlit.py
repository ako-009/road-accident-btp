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
    page_title="Road Accident Analysis - India",
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
    st.markdown("**Data source:** iRAD dataset")
    st.markdown("**Model:** Poisson Regression")
    st.markdown("**Framework:** 4E (Campbell Review)")
    st.markdown("---")
    st.markdown("**Project:** BTP Road Safety")
    st.markdown("**Institute:** IIT Kharagpur")


tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "EDA Plots", "Analysis", "AI Chat"
])


# ── TAB 1: OVERVIEW ───────────────────────────────────────────────
with tab1:
    st.markdown("## National Road Accident Overview")
    st.markdown(
        "Data from the **Integrated Road Accident Database (iRAD)** "
        "covering all 36 states and union territories of India."
    )
    totals = api_get("/totals")
    if totals.get("status") == "ok":
        year = totals.get("latest_year", "N/A")
        st.markdown(f"#### Statistics for {year}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Accidents",   f"{totals['total_accidents']:,}")
        c2.metric("Total Deaths",      f"{totals['total_killed']:,}")
        c3.metric("Grievous Injuries", f"{totals['total_grievous']:,}",
                  help="2023 data — breakdown available from 2023 onwards")
        c4.metric("Avg Fatality Rate", f"{totals['avg_fatality_rate']:.1f}%")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("States Covered", int(totals["states_covered"]))
        c6.metric("Minor Injuries", f"{totals['total_minor']:,}",
                  help="Persons with minor injuries in road accidents (2023)")
        c7.metric("Years of Data",  len(totals.get("years_in_data", [])))
        c8.metric("Data Period",    f"2019 - {year}")
    else:
        st.warning("Could not load totals. Make sure the API is running.")

    st.markdown("---")
    st.markdown("### Top 10 States by Deaths (2023)")
    top = api_get("/top-states", params={"by": "killed", "year": 2023, "n": 10})
    if top.get("status") == "ok":
        df_top = pd.DataFrame(top["top_states"])
        df_top.columns = [
            "State", "Total Accidents", "Deaths",
            "Fatality Rate (%)", "Severity Index"
        ]
        df_top.index = range(1, len(df_top) + 1)
        st.dataframe(df_top, use_container_width=True)

    st.markdown("---")
    st.markdown("### About iRAD and the 4E Framework")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
The **iRAD** was launched on Jan 13, 2020 by MoRTH.
Developed by NIC, it covers all 36 states and UTs.

**Key features:**
- Mobile-based on-site accident data collection
- Black spot identification using GIS grid analysis
- eDAR for cashless victim treatment
- Integration with CCTNS, Vahan, Sarathi, NHA, eCourts
        """)
    with col_b:
        st.markdown("""
**The 4E Framework (Campbell Systematic Review):**

| E | Description |
|---|---|
| Engineering | Road design, barriers, signage, lighting |
| Enforcement | Speed cameras, drunk-driving checks |
| Education | Campaigns, driver training |
| Emergency | Ambulance response, trauma centres |

Evidence shows combining all 4Es reduces fatalities by 30-50%.
        """)


# ── TAB 2: EDA PLOTS ──────────────────────────────────────────────
with tab2:
    st.markdown("## Exploratory Data Analysis")
    st.markdown("All charts generated from the cleaned iRAD dataset.")

    plot_data  = api_get("/plots")
    plot_files = plot_data.get("plots", [])
    if not plot_files:
        plot_files = sorted([p.name for p in PLOTS_DIR.glob("*.png")])

    if not plot_files:
        st.warning("No plots found. Run python src/eda.py first.")
    else:
        descriptions = {
            "01_top_accidents_by_state.png": "Top 10 states by total accidents (2022).",
            "02_top_deaths_by_state.png":    "Top 10 states by road accident deaths (2022).",
            "03_fatality_rate_by_state.png": "States with highest fatality rate. Red line = national average.",
            "04_yearly_national_trend.png":  "Year-over-year national trend 2019-2023. COVID dip in 2020.",
            "05_severity_breakdown.png":     "Accident severity breakdown for top 10 states (2023).",
            "06_covid_impact.png":           "COVID-19 impact. 2020 shows sharp drop due to lockdowns.",
            "07_day_vs_night.png":           "Day vs Night accidents — Top 10 states (2022). Night = 18:00 to 06:00 hrs.",
            "08_road_type_accidents.png":    "Change in fatality rate per state: 2019 vs 2023. Green = improved, Red = worsened.",
            "09_actual_vs_predicted.png":    "Poisson model: actual vs predicted. Points near diagonal = good fit.",
            "10_feature_importance.png":     "Model coefficients showing which features influence accident counts.",
            "11_blackspot_risk_map.png":     "Black spot risk map: Critical, High, Medium, Low risk states.",
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
            "States in top 25% nationally on 2 or more criteria are classified as black spots."
        )
        risk_filter = st.selectbox(
            "Filter by risk level",
            ["all", "Critical", "High", "Medium", "Low"]
        )
        bs_data = api_get("/blackspots", params={"risk_level": risk_filter})
        if bs_data.get("status") == "ok" and bs_data.get("blackspots"):
            df_bs = pd.DataFrame(bs_data["blackspots"])
            st.dataframe(
                df_bs[["state", "total_accidents", "killed",
                        "fatality_rate", "risk_level"]].rename(columns={
                    "state":         "State",
                    "total_accidents": "Accidents",
                    "killed":        "Deaths",
                    "fatality_rate": "Fatality Rate (%)",
                    "risk_level":    "Risk Level",
                }),
                use_container_width=True,
            )
            st.markdown("#### Recommended Interventions")
            selected = st.selectbox("Select a state", df_bs["state"].tolist())
            if selected:
                row = df_bs[df_bs["state"] == selected].iloc[0]
                for item in row.get("recommended_interventions", "").split(" | "):
                    if item.strip():
                        parts = item.split(":", 1)
                        if len(parts) == 2:
                            cat, detail = parts
                            icons = {
                                "Engineering": "[Engineering]",
                                "Enforcement": "[Enforcement]",
                                "Education":   "[Education]",
                                "Emergency":   "[Emergency]",
                            }
                            label = icons.get(cat.strip(), "[•]")
                            st.markdown(f"{label} **{cat.strip()}:** {detail.strip()}")
        else:
            st.info("No black spots found for the selected filter.")

    with sec2:
        st.markdown("### National Year-over-Year Trend")
        trend_data = api_get("/trend")
        if trend_data.get("status") == "ok":
            df_trend = pd.DataFrame(trend_data["trend"])
            df_trend.columns = [
                "Year", "Total Accidents", "Total Deaths", "Avg Fatality Rate (%)"
            ]
            st.dataframe(df_trend.set_index("Year"), use_container_width=True)
            st.line_chart(
                df_trend[["Year", "Total Accidents", "Total Deaths"]].set_index("Year")
            )
            st.info(
                "Drop in 2020 = COVID-19 lockdowns. "
                "Accidents rebounded in 2021-2023 as traffic normalised."
            )
        else:
            st.warning("Could not load trend data.")

    with sec3:
        st.markdown("### Poisson Regression Model Performance")
        st.markdown(
            "Poisson regression predicts accident counts. Designed for count data "
            "(whole non-negative numbers), used in official road safety research worldwide."
        )
        metrics_data = api_get("/metrics")
        if metrics_data.get("status") == "ok":
            m = metrics_data["metrics"]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MAE",  f"{float(m['mae']):,.0f} accidents",
                       help="Mean Absolute Error")
            mc2.metric("RMSE", f"{float(m['rmse']):,.0f} accidents",
                       help="Root Mean Square Error")
            mc3.metric("MAPE", f"{float(m['mape_pct']):.1f}%",
                       help="Mean Absolute Percentage Error")
            st.markdown("**Features used:**")
            for feat in str(m.get("features_used", "")).split(", "):
                st.markdown(f"- `{feat.strip()}`")
        else:
            st.warning("Run python src/train_poisson.py first.")


# ── TAB 4: AI CHAT ────────────────────────────────────────────────
with tab4:
    st.markdown("## AI Road Safety Assistant")
    st.markdown(
        "Ask anything about road accidents in India. "
        "The assistant searches the knowledge base and answers using OpenAI."
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
        with st.spinner("Searching and generating answer..."):
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
                f'<div style="background:#e8f4f8;padding:10px;border-radius:8px;'
                f'margin-bottom:6px"><b>You:</b> {entry["question"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#f0f7f0;padding:10px;border-radius:8px;'
                f'margin-bottom:4px"><b>Assistant:</b> {entry["answer"]}</div>',
                unsafe_allow_html=True,
            )
            if entry.get("sources"):
                with st.expander(f"Sources ({len(entry['sources'])} documents)"):
                    for j, src in enumerate(entry["sources"], 1):
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
Campbell Review interventions, and model predictions.

**Example questions:**
- Which state has the highest fatality rate?
- What engineering interventions reduce road deaths?
- How did COVID affect road accidents in India?
- Tell me about road accidents in Rajasthan
        """)