"""
tools.py
--------
Tool functions for the Road Accident AI Agent.
All functions query the SQLite database (data/road_accidents.db).
Falls back to CSV if database not found.
"""

import sqlite3
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "road_accidents.db"


def get_db():
    """Returns a SQLite connection. Falls back to CSV mode if DB not found."""
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    return None


def _df_from_csv():
    """Fallback: load from cleaned CSV if database not available."""
    csv_path = ROOT / "data" / "processed" / "accidents_cleaned.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


# ── Tool 1: National Totals ───────────────────────────────────────
def get_national_totals():
    """Returns national accident totals for the latest year."""
    conn = get_db()
    try:
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(year) FROM accidents")
            latest_year = cursor.fetchone()[0]

            cursor.execute("""
                SELECT
                    SUM(total_accidents) as total_accidents,
                    SUM(killed)          as total_killed,
                    SUM(total_injured)   as total_injured,
                    SUM(grievous_injury) as total_grievous,
                    SUM(minor_injured)   as total_minor,
                    AVG(fatality_rate)   as avg_fatality_rate,
                    COUNT(DISTINCT state) as states_covered
                FROM accidents
                WHERE year = ?
            """, (latest_year,))
            row = cursor.fetchone()

            cursor.execute("SELECT DISTINCT year FROM accidents ORDER BY year")
            years = [r[0] for r in cursor.fetchall()]
            conn.close()

            return {
                "status":            "ok",
                "latest_year":       latest_year,
                "total_accidents":   int(row["total_accidents"] or 0),
                "total_killed":      int(row["total_killed"] or 0),
                "total_injured":     int(row["total_injured"] or 0),
                "total_grievous":    int(row["total_grievous"] or 0),
                "total_minor":       int(row["total_minor"] or 0),
                "avg_fatality_rate": round(float(row["avg_fatality_rate"] or 0), 2),
                "states_covered":    int(row["states_covered"] or 0),
                "years_in_data":     years,
            }
        else:
            # CSV fallback
            df = _df_from_csv()
            latest_year = int(df["year"].max())
            df_year = df[df["year"] == latest_year]
            return {
                "status":            "ok",
                "latest_year":       latest_year,
                "total_accidents":   int(df_year["total_accidents"].sum()),
                "total_killed":      int(df_year["killed"].sum()),
                "total_injured":     int(df_year.get("total_injured", pd.Series([0])).sum()),
                "total_grievous":    int(df_year.get("grievous_injury", pd.Series([0])).sum()),
                "total_minor":       int(df_year.get("minor_injured", pd.Series([0])).sum()),
                "avg_fatality_rate": round(float(df_year["fatality_rate"].mean()), 2),
                "states_covered":    int(df["state"].nunique()),
                "years_in_data":     sorted(df["year"].unique().tolist()),
            }
    except Exception as e:
        if conn: conn.close()
        return {"status": "error", "message": str(e)}


# ── Tool 2: Top States ────────────────────────────────────────────
def get_top_states(by: str = "killed", year: int = 2022, n: int = 10):
    """Returns top N states by a given metric for a given year."""
    valid_cols = {
        "killed":         "killed",
        "total_accidents":"total_accidents",
        "fatality_rate":  "fatality_rate",
        "night_accidents":"night_accidents",
    }
    col = valid_cols.get(by, "killed")

    conn = get_db()
    try:
        if conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT state, total_accidents, killed, fatality_rate,
                       ROUND(CAST(killed AS REAL) / NULLIF(total_accidents,0), 4) as severity_index
                FROM accidents
                WHERE year = ?
                ORDER BY {col} DESC
                LIMIT ?
            """, (year, n))
            rows = cursor.fetchall()
            conn.close()

            return {
                "status": "ok",
                "year":   year,
                "by":     by,
                "top_states": [{
                    "state":           r["state"],
                    "total_accidents": int(r["total_accidents"] or 0),
                    "killed":          int(r["killed"] or 0),
                    "fatality_rate":   round(float(r["fatality_rate"] or 0), 2),
                    "severity_index":  round(float(r["severity_index"] or 0), 4),
                } for r in rows],
            }
        else:
            df = _df_from_csv()
            df_year = df[df["year"] == year].copy()
            df_year["severity_index"] = df_year.apply(
                lambda r: round(r["killed"] / r["total_accidents"], 4)
                if r["total_accidents"] > 0 else 0, axis=1
            )
            top = df_year.nlargest(n, col)
            return {
                "status": "ok",
                "year":   year,
                "by":     by,
                "top_states": top[[
                    "state","total_accidents","killed","fatality_rate","severity_index"
                ]].to_dict("records"),
            }
    except Exception as e:
        if conn: conn.close()
        return {"status": "error", "message": str(e)}


# ── Tool 3: State Summary ─────────────────────────────────────────
def get_state_summary(state_name: str):
    """Returns detailed summary for one state."""
    conn = get_db()
    try:
        if conn:
            cursor = conn.cursor()

            # Get summary row
            cursor.execute("""
                SELECT * FROM states WHERE LOWER(state) = LOWER(?)
            """, (state_name,))
            summary = cursor.fetchone()

            if not summary:
                # Try partial match
                cursor.execute("""
                    SELECT * FROM states WHERE LOWER(state) LIKE LOWER(?)
                """, (f"%{state_name}%",))
                summary = cursor.fetchone()

            if not summary:
                conn.close()
                return {"status": "error", "message": f"State '{state_name}' not found"}

            # Get yearly breakdown
            cursor.execute("""
                SELECT year, total_accidents, killed, fatality_rate,
                       night_accidents, night_share, nh_accidents
                FROM accidents
                WHERE LOWER(state) = LOWER(?)
                ORDER BY year
            """, (summary["state"],))
            yearly = cursor.fetchall()

            # Find worst year
            cursor.execute("""
                SELECT year FROM accidents
                WHERE LOWER(state) = LOWER(?)
                ORDER BY total_accidents DESC LIMIT 1
            """, (summary["state"],))
            worst = cursor.fetchone()
            conn.close()

            return {
                "status":            "ok",
                "state":             summary["state"],
                "total_accidents":   int(summary["total_accidents"] or 0),
                "total_killed":      int(summary["total_killed"] or 0),
                "avg_fatality_rate": round(float(summary["avg_fatality_rate"] or 0), 2),
                "worst_year":        int(worst["year"]) if worst else "N/A",
                "yearly_breakdown":  [{
                    "year":         int(r["year"]),
                    "accidents":    int(r["total_accidents"] or 0),
                    "killed":       int(r["killed"] or 0),
                    "fatality_rate":round(float(r["fatality_rate"] or 0), 2),
                    "night_accidents": int(r["night_accidents"] or 0),
                    "night_share":  round(float(r["night_share"] or 0), 2),
                } for r in yearly],
            }
        else:
            df = _df_from_csv()
            df_state = df[df["state"].str.lower() == state_name.lower()]
            if df_state.empty:
                df_state = df[df["state"].str.lower().str.contains(state_name.lower())]
            if df_state.empty:
                return {"status": "error", "message": f"State '{state_name}' not found"}

            worst_year = int(df_state.loc[df_state["total_accidents"].idxmax(), "year"])
            return {
                "status":            "ok",
                "state":             df_state["state"].iloc[0],
                "total_accidents":   int(df_state["total_accidents"].sum()),
                "total_killed":      int(df_state["killed"].sum()),
                "avg_fatality_rate": round(float(df_state["fatality_rate"].mean()), 2),
                "worst_year":        worst_year,
                "yearly_breakdown":  df_state.sort_values("year")[[
                    "year","total_accidents","killed","fatality_rate"
                ]].rename(columns={"total_accidents":"accidents"}).to_dict("records"),
            }
    except Exception as e:
        if conn: conn.close()
        return {"status": "error", "message": str(e)}


# ── Tool 4: Black Spots ───────────────────────────────────────────
def get_blackspots(risk_level: str = "all"):
    """Returns black spot states filtered by risk level."""
    conn = get_db()
    try:
        if conn:
            cursor = conn.cursor()
            if risk_level.lower() == "all":
                cursor.execute("""
                    SELECT * FROM blackspots
                    ORDER BY CASE risk_level
                        WHEN 'Critical' THEN 1
                        WHEN 'High'     THEN 2
                        WHEN 'Medium'   THEN 3
                        ELSE 4 END, fatality_rate DESC
                """)
            else:
                cursor.execute("""
                    SELECT * FROM blackspots WHERE risk_level = ?
                    ORDER BY fatality_rate DESC
                """, (risk_level,))
            rows = cursor.fetchall()
            conn.close()

            return {
                "status":     "ok",
                "risk_level": risk_level,
                "blackspots": [{
                    "state":                      r["state"],
                    "total_accidents":            int(r["total_accidents"] or 0),
                    "killed":                     int(r["killed"] or 0),
                    "fatality_rate":              round(float(r["fatality_rate"] or 0), 2),
                    "risk_level":                 r["risk_level"],
                    "recommended_interventions":  r["recommended_interventions"],
                } for r in rows],
            }
        else:
            csv_path = ROOT / "outputs" / "blackspots.csv"
            df = pd.read_csv(csv_path)
            if risk_level.lower() != "all":
                df = df[df["risk_level"] == risk_level]
            return {"status": "ok", "risk_level": risk_level, "blackspots": df.to_dict("records")}
    except Exception as e:
        if conn: conn.close()
        return {"status": "error", "message": str(e)}


# ── Tool 5: Plot List ─────────────────────────────────────────────
def get_plot_list():
    """Returns list of available plot filenames."""
    plots_dir = ROOT / "outputs" / "plots"
    if plots_dir.exists():
        plots = sorted([p.name for p in plots_dir.glob("*.png")])
        return {"status": "ok", "plots": plots, "count": len(plots)}
    return {"status": "ok", "plots": [], "count": 0}


# ── Tool 6: Model Metrics ─────────────────────────────────────────
def get_model_metrics():
    """Returns Poisson model performance metrics."""
    conn = get_db()
    try:
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT metric_name, metric_value FROM model_metrics")
            rows = cursor.fetchall()
            conn.close()

            if rows:
                metrics = {r["metric_name"]: r["metric_value"] for r in rows}

                # Get features from predictions table
                csv_path = ROOT / "outputs" / "model_metrics.csv"
                features_used = "is_covid_year, is_large_state, nh_share, night_share, fatality_rate, severity_index"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    if "features_used" in df.columns:
                        features_used = df["features_used"].iloc[0]

                return {
                    "status": "ok",
                    "metrics": {
                        "mae":           metrics.get("mae", 0),
                        "rmse":          metrics.get("rmse", 0),
                        "mape_pct":      metrics.get("mape_pct", 0),
                        "train_rows":    int(metrics.get("train_rows", 0)),
                        "test_rows":     int(metrics.get("test_rows", 0)),
                        "features_used": features_used,
                    }
                }

        # CSV fallback
        csv_path = ROOT / "outputs" / "model_metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            row = df.iloc[0]
            return {
                "status": "ok",
                "metrics": {
                    "mae":          float(row.get("mae", 0)),
                    "rmse":         float(row.get("rmse", 0)),
                    "mape_pct":     float(row.get("mape_pct", 0)),
                    "train_rows":   int(row.get("train_rows", 0)),
                    "test_rows":    int(row.get("test_rows", 0)),
                    "features_used":str(row.get("features_used", "")),
                }
            }
        return {"status": "error", "message": "Metrics not found"}
    except Exception as e:
        if conn: conn.close()
        return {"status": "error", "message": str(e)}


# ── Tool 7: Yearly Trend ──────────────────────────────────────────
def get_yearly_trend():
    """Returns year-over-year national trend."""
    conn = get_db()
    try:
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    year,
                    SUM(total_accidents) as total_accidents,
                    SUM(killed)          as total_killed,
                    AVG(fatality_rate)   as avg_fatality_rate
                FROM accidents
                GROUP BY year
                ORDER BY year
            """)
            rows = cursor.fetchall()
            conn.close()

            return {
                "status": "ok",
                "trend":  [{
                    "year":             int(r["year"]),
                    "total_accidents":  int(r["total_accidents"] or 0),
                    "total_killed":     int(r["total_killed"] or 0),
                    "avg_fatality_rate":round(float(r["avg_fatality_rate"] or 0), 2),
                } for r in rows],
            }
        else:
            df = _df_from_csv()
            trend = df.groupby("year").agg(
                total_accidents=("total_accidents","sum"),
                total_killed=("killed","sum"),
                avg_fatality_rate=("fatality_rate","mean"),
            ).round(2).reset_index()
            return {"status": "ok", "trend": trend.to_dict("records")}
    except Exception as e:
        if conn: conn.close()
        return {"status": "error", "message": str(e)}


# ── Tool 8: Database Stats (new!) ─────────────────────────────────
def get_database_stats():
    """Returns database statistics — new endpoint."""
    conn = get_db()
    if not conn:
        return {"status": "error", "message": "Database not found"}
    try:
        cursor = conn.cursor()
        stats = {}

        for table in ["accidents", "states", "blackspots", "model_metrics", "predictions"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(year), MAX(year) FROM accidents")
        row = cursor.fetchone()
        stats["year_range"] = f"{row[0]} - {row[1]}"

        cursor.execute("SELECT COUNT(DISTINCT state) FROM accidents")
        stats["states_covered"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM accidents WHERE night_accidents > 0")
        stats["rows_with_night_data"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM accidents WHERE nh_accidents > 0")
        stats["rows_with_nh_data"] = cursor.fetchone()[0]

        conn.close()
        return {
            "status": "ok",
            "database_path": str(DB_PATH),
            "tables": stats,
        }
    except Exception as e:
        conn.close()
        return {"status": "error", "message": str(e)}
