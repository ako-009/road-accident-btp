"""
setup_database.py
-----------------
Creates and populates the SQLite database from CSV files.
Run this once to set up the database:
    python src/setup_database.py

Database: data/road_accidents.db
Tables:
    - accidents       : state-year accident data
    - states          : state summary statistics
    - blackspots      : black spot classifications
    - model_metrics   : Poisson model performance
    - predictions     : model predictions vs actuals
"""

import sqlite3
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "road_accidents.db"


def get_connection():
    """Returns a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)


def create_tables(conn):
    """Creates all tables with proper schema."""
    cursor = conn.cursor()

    cursor.executescript("""
        -- Main accidents table
        CREATE TABLE IF NOT EXISTS accidents (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            state            TEXT NOT NULL,
            year             INTEGER NOT NULL,
            total_accidents  INTEGER DEFAULT 0,
            killed           INTEGER DEFAULT 0,
            total_injured    INTEGER DEFAULT 0,
            fatal_accidents  INTEGER DEFAULT 0,
            grievous_injury  INTEGER DEFAULT 0,
            minor_hosp       INTEGER DEFAULT 0,
            no_injury        INTEGER DEFAULT 0,
            grievously_injured INTEGER DEFAULT 0,
            minor_injured    INTEGER DEFAULT 0,
            fatality_rate    REAL DEFAULT 0.0,
            night_accidents  INTEGER DEFAULT 0,
            night_share      REAL DEFAULT 0.0,
            nh_accidents     INTEGER DEFAULT 0,
            nh_share         REAL DEFAULT 0.0,
            sh_accidents     INTEGER DEFAULT 0,
            UNIQUE(state, year)
        );

        -- State summary table
        CREATE TABLE IF NOT EXISTS states (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            state             TEXT UNIQUE NOT NULL,
            total_accidents   INTEGER DEFAULT 0,
            total_killed      INTEGER DEFAULT 0,
            avg_fatality_rate REAL DEFAULT 0.0,
            years_of_data     INTEGER DEFAULT 0,
            total_grievous    INTEGER DEFAULT 0,
            total_minor       INTEGER DEFAULT 0,
            avg_severity      REAL DEFAULT 0.0,
            night_share       REAL DEFAULT 0.0,
            nh_share          REAL DEFAULT 0.0
        );

        -- Black spots table
        CREATE TABLE IF NOT EXISTS blackspots (
            id                        INTEGER PRIMARY KEY AUTOINCREMENT,
            state                     TEXT UNIQUE NOT NULL,
            total_accidents           INTEGER DEFAULT 0,
            killed                    INTEGER DEFAULT 0,
            fatality_rate             REAL DEFAULT 0.0,
            risk_level                TEXT DEFAULT 'Low',
            recommended_interventions TEXT DEFAULT ''
        );

        -- Model metrics table
        CREATE TABLE IF NOT EXISTS model_metrics (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name   TEXT UNIQUE NOT NULL,
            metric_value  REAL NOT NULL,
            description   TEXT DEFAULT ''
        );

        -- Predictions table
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            state       TEXT NOT NULL,
            year        INTEGER NOT NULL,
            actual      REAL DEFAULT 0,
            predicted   REAL DEFAULT 0,
            error       REAL DEFAULT 0,
            UNIQUE(state, year)
        );
    """)
    conn.commit()
    print("  Tables created successfully.")


def load_accidents(conn):
    """Load accidents data from cleaned CSV."""
    csv_path = ROOT / "data" / "processed" / "accidents_cleaned.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping accidents table.")
        return

    df = pd.read_csv(csv_path)

    # Ensure all required columns exist
    required = [
        "state", "year", "total_accidents", "killed", "total_injured",
        "fatal_accidents", "grievous_injury", "minor_hosp", "no_injury",
        "grievously_injured", "minor_injured", "fatality_rate",
        "night_accidents", "nh_accidents", "sh_accidents",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = 0

    # Add night_share and nh_share if not present
    if "night_share" not in df.columns:
        df["night_share"] = df.apply(
            lambda r: round(r["night_accidents"] / r["total_accidents"] * 100, 2)
            if r["total_accidents"] > 0 and r["night_accidents"] > 0 else 0.0, axis=1
        )
    if "nh_share" not in df.columns:
        df["nh_share"] = df.apply(
            lambda r: round(r["nh_accidents"] / r["total_accidents"] * 100, 2)
            if r["total_accidents"] > 0 and r["nh_accidents"] > 0 else 0.0, axis=1
        )

    cursor = conn.cursor()
    cursor.execute("DELETE FROM accidents")

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO accidents (
                state, year, total_accidents, killed, total_injured,
                fatal_accidents, grievous_injury, minor_hosp, no_injury,
                grievously_injured, minor_injured, fatality_rate,
                night_accidents, night_share, nh_accidents, nh_share, sh_accidents
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            str(row["state"]), int(row["year"]),
            int(row["total_accidents"]), int(row["killed"]),
            int(row.get("total_injured", 0)),
            int(row.get("fatal_accidents", 0)),
            int(row.get("grievous_injury", 0)),
            int(row.get("minor_hosp", 0)),
            int(row.get("no_injury", 0)),
            int(row.get("grievously_injured", 0)),
            int(row.get("minor_injured", 0)),
            float(row.get("fatality_rate", 0.0)),
            int(row.get("night_accidents", 0)),
            float(row.get("night_share", 0.0)),
            int(row.get("nh_accidents", 0)),
            float(row.get("nh_share", 0.0)),
            int(row.get("sh_accidents", 0)),
        ))

    conn.commit()
    print(f"  Loaded {len(df)} rows into accidents table.")


def load_states(conn):
    """Load state summary from summary CSV."""
    csv_path = ROOT / "outputs" / "summary_by_state.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping states table.")
        return

    df = pd.read_csv(csv_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM states")

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO states (
                state, total_accidents, total_killed,
                avg_fatality_rate, years_of_data,
                total_grievous, total_minor, avg_severity,
                night_share, nh_share
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            str(row["state"]),
            int(row.get("total_accidents", 0)),
            int(row.get("total_killed", 0)),
            float(row.get("avg_fatality_rate", 0.0)),
            int(row.get("years_of_data", 5)),
            int(row.get("total_grievous", 0)),
            int(row.get("total_minor", 0)),
            float(row.get("avg_severity", 0.0)),
            float(row.get("night_share", 0.0)),
            float(row.get("nh_share", 0.0)),
        ))

    conn.commit()
    print(f"  Loaded {len(df)} rows into states table.")


def load_blackspots(conn):
    """Load blackspot data."""
    csv_path = ROOT / "outputs" / "blackspots.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping blackspots table.")
        return

    df = pd.read_csv(csv_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM blackspots")

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO blackspots (
                state, total_accidents, killed,
                fatality_rate, risk_level, recommended_interventions
            ) VALUES (?,?,?,?,?,?)
        """, (
            str(row["state"]),
            int(row.get("total_accidents", 0)),
            int(row.get("killed", 0)),
            float(row.get("fatality_rate", 0.0)),
            str(row.get("risk_level", "Low")),
            str(row.get("recommended_interventions", "")),
        ))

    conn.commit()
    print(f"  Loaded {len(df)} rows into blackspots table.")


def load_metrics(conn):
    """Load model metrics."""
    csv_path = ROOT / "outputs" / "model_metrics.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping metrics table.")
        return

    df = pd.read_csv(csv_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM model_metrics")

    descriptions = {
        "mae":        "Mean Absolute Error — average prediction error per state per year",
        "rmse":       "Root Mean Square Error — penalises large errors more",
        "mape_pct":   "Mean Absolute Percentage Error — relative prediction error",
        "train_rows": "Number of rows used for training",
        "test_rows":  "Number of rows used for testing",
    }

    metric_cols = [c for c in descriptions.keys() if c in df.columns]
    row = df.iloc[0]  # single-row wide format

    for name in metric_cols:
        val = float(row[name])
        cursor.execute("""
            INSERT OR REPLACE INTO model_metrics (metric_name, metric_value, description)
            VALUES (?,?,?)
        """, (name, val, descriptions.get(name, "")))

    conn.commit()
    print(f"  Loaded {len(metric_cols)} rows into model_metrics table.")


def load_predictions(conn):
    """Load model predictions."""
    csv_path = ROOT / "outputs" / "poisson_predictions.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping predictions table.")
        return

    df = pd.read_csv(csv_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")

    for _, row in df.iterrows():
        actual    = float(row.get("actual", row.get("y_actual", 0)))
        predicted = float(row.get("predicted", row.get("y_pred", 0)))
        cursor.execute("""
            INSERT OR REPLACE INTO predictions (state, year, actual, predicted, error)
            VALUES (?,?,?,?,?)
        """, (
            str(row.get("state", "")),
            int(row.get("year", 0)),
            actual,
            predicted,
            round(abs(actual - predicted), 2),
        ))

    conn.commit()
    print(f"  Loaded {len(df)} rows into predictions table.")


def print_summary(conn):
    """Print summary of database contents."""
    cursor = conn.cursor()
    print("\n  Database Summary:")
    print("  " + "─" * 40)
    for table in ["accidents", "states", "blackspots", "model_metrics", "predictions"]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table:<20} {count:>6} rows")
    print("  " + "─" * 40)
    print(f"  Database saved at: {DB_PATH}")


def main():
    print("=== Setting up SQLite Database ===")
    print(f"  Database path: {DB_PATH}")

    conn = get_connection()

    print("\n  Creating tables...")
    create_tables(conn)

    print("\n  Loading data...")
    load_accidents(conn)
    load_states(conn)
    load_blackspots(conn)
    load_metrics(conn)
    load_predictions(conn)

    print_summary(conn)
    conn.close()
    print("\n=== Database setup complete! ===")


if __name__ == "__main__":
    main()
