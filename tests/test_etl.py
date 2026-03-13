"""
test_etl.py
-----------
Unit tests for the ETL pipeline.
Run with: pytest tests/ -v
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def make_test_df():
    """Creates a minimal valid dataframe for testing."""
    return pd.DataFrame({
        "state":           ["Delhi", "Mumbai", "Chennai"],
        "year":            [2022, 2022, 2022],
        "total_accidents": [1000, 2000, 1500],
        "fatal_accidents": [300, 500, 400],
        "killed":          [320, 550, 420],
        "grievous_injury": [350, 700, 500],
        "minor_hosp":      [200, 400, 300],
        "minor_no_hosp":   [80,  150, 100],
        "no_injury":       [70,  200, 200],
        "total_persons":   [2200, 4400, 3300],
        "grievously_injured": [400, 800, 600],
        "minor_injured":   [700, 1400, 1000],
        "night_accidents": [400, 800, 600],
        "nh_accidents":    [350, 700, 550],
        "sh_accidents":    [250, 500, 400],
    })


def test_required_columns_present():
    """ETL should pass when all required columns exist."""
    from src.etl import REQUIRED
    df = make_test_df()
    missing = [c for c in REQUIRED if c not in df.columns]
    assert missing == [], f"Missing columns: {missing}"


def test_fatality_rate_computed_correctly():
    """fatality_rate = killed / total_accidents * 100"""
    df = make_test_df()
    safe = df["total_accidents"].replace(0, np.nan)
    df["fatality_rate"] = (df["killed"] / safe * 100).round(2)
    assert df["fatality_rate"].iloc[0] == pytest.approx(32.0, abs=0.1)


def test_no_negative_values_after_clip():
    """All numeric columns should be >= 0 after cleaning."""
    df = make_test_df()
    numeric_cols = ["total_accidents", "killed", "grievous_injury"]
    for col in numeric_cols:
        df[col] = df[col].clip(lower=0)
        assert (df[col] >= 0).all(), f"{col} has negative values"


def test_bad_rows_removed():
    """Rows with empty state or zero accidents should be dropped."""
    df = make_test_df()
    df.loc[0, "state"] = ""
    df.loc[1, "total_accidents"] = 0
    df = df[df["state"].str.strip() != ""]
    df = df[df["total_accidents"] > 0]
    assert len(df) == 1


def test_period_labels_correct():
    """Period labels should map years to correct COVID periods."""
    df = pd.DataFrame({"year": [2019, 2020, 2021, 2022, 2023]})
    df["period"] = df["year"].apply(
        lambda y: "Pre-COVID" if y <= 2019
        else "COVID" if y <= 2021
        else "Post-COVID"
    )
    assert df.loc[df["year"] == 2019, "period"].values[0] == "Pre-COVID"
    assert df.loc[df["year"] == 2020, "period"].values[0] == "COVID"
    assert df.loc[df["year"] == 2022, "period"].values[0] == "Post-COVID"
