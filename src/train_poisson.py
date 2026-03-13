"""
train_poisson.py
-----------------
Trains a Poisson regression model to predict accident counts.

Why Poisson regression?
  - Accident counts are whole numbers (0, 1, 2, ...)
  - They cannot be negative
  - Poisson regression is specifically designed for count data
  - Used in official road safety research worldwide (see Campbell review)

Run with:
    python src/train_poisson.py
"""

import sys
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (OUTPUT_DIR, PLOTS_DIR, POISSON_MODEL_FILE,
                        PREDICTIONS_CSV, ensure_dirs)

FEATURES_CSV = OUTPUT_DIR / "features.csv"
METRICS_CSV  = OUTPUT_DIR / "model_metrics.csv"

# These are the input columns we feed into the model
FEATURE_COLS = [
    "is_covid_year",
    "is_large_state",
    "nh_share",
    "night_share",
    "fatality_rate",
    "severity_index",
]

TARGET_COL = "total_accidents"


def load_features() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Features CSV not found. Run python src/features.py first."
        )
    return pd.read_csv(FEATURES_CSV)


def train_model(df: pd.DataFrame):
    """
    Trains the Poisson regression model and returns:
      - trained model
      - scaler (for transforming new inputs)
      - predictions dataframe
      - metrics dictionary
    """

    # ── Prepare X (inputs) and y (target) ─────────────────────
    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]

    # ── Split: 80% train, 20% test ────────────────────────────
    # random_state=42 ensures we get the same split every time
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Training set : {len(X_train)} rows")
    print(f"  Test set     : {len(X_test)} rows")

    # ── Scale features ────────────────────────────────────────
    # StandardScaler makes all features have mean=0, std=1
    # This helps the model converge faster and more reliably
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Train Poisson model ───────────────────────────────────
    model = PoissonRegressor(alpha=0.1, max_iter=500)
    model.fit(X_train_scaled, y_train)

    # ── Predict ───────────────────────────────────────────────
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test  = model.predict(X_test_scaled)

    # ── Evaluate ──────────────────────────────────────────────
    mae  = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    # Mean Absolute Percentage Error
    mape = (np.abs((y_test - y_pred_test) / y_test.replace(0, np.nan))
            .fillna(0).mean() * 100)

    metrics = {
        "mae":          round(mae,  2),
        "rmse":         round(rmse, 2),
        "mape_pct":     round(mape, 2),
        "train_rows":   len(X_train),
        "test_rows":    len(X_test),
        "features_used": ", ".join(FEATURE_COLS),
    }
    print(f"\n  Model performance on test set:")
    print(f"    MAE  (mean absolute error) : {mae:,.0f} accidents")
    print(f"    RMSE (root mean sq error)  : {rmse:,.0f} accidents")
    print(f"    MAPE (mean abs % error)    : {mape:.1f}%")

    # ── Build predictions dataframe ───────────────────────────
    pred_df = df[["state", "year", TARGET_COL]].copy()
    pred_df["predicted_accidents"] = model.predict(
        scaler.transform(X.fillna(0))
    ).round(0).astype(int)
    pred_df["residual"] = pred_df[TARGET_COL] - pred_df["predicted_accidents"]
    pred_df["abs_error"] = pred_df["residual"].abs()

    return model, scaler, pred_df, metrics


def plot_actual_vs_predicted(pred_df: pd.DataFrame):
    """Scatter plot of actual vs predicted — good model = points near diagonal."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pred_df["total_accidents"], pred_df["predicted_accidents"],
               alpha=0.5, color="steelblue", edgecolors="white", s=60)
    # Perfect prediction line
    lim = max(pred_df["total_accidents"].max(),
              pred_df["predicted_accidents"].max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Accidents")
    ax.set_ylabel("Predicted Accidents")
    ax.set_title("Poisson Model: Actual vs Predicted Accidents",
                 fontweight="bold")
    ax.legend(fontsize=9)
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    path = PLOTS_DIR / "09_actual_vs_predicted.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot  ->  {path.name}")


def plot_feature_importance(model, feature_names: list):
    """Bar chart of model coefficients — shows which features matter most."""
    coefs = pd.Series(model.coef_, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d73027" if c < 0 else "#1a9850" for c in coefs]
    ax.barh(coefs.index, coefs.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (effect on accident count)")
    ax.set_title("Poisson Model: Feature Coefficients", fontweight="bold")
    ax.annotate("Green = increases accidents\nRed = decreases accidents",
                xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    path = PLOTS_DIR / "10_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot  ->  {path.name}")


def run_training() -> dict:
    ensure_dirs()
    print("\n=== Poisson Regression Training ===")
    df = load_features()
    print(f"  Loaded {len(df)} rows with features")

    model, scaler, pred_df, metrics = train_model(df)

    # Save model and scaler together in one file
    joblib.dump({"model": model, "scaler": scaler,
                 "feature_cols": FEATURE_COLS}, POISSON_MODEL_FILE)
    print(f"\n  Model saved  ->  {POISSON_MODEL_FILE}")

    # Save predictions
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"  Predictions  ->  {PREDICTIONS_CSV}")

    # Save metrics
    pd.DataFrame([metrics]).to_csv(METRICS_CSV, index=False)
    print(f"  Metrics      ->  {METRICS_CSV}")

    # Save plots
    plot_actual_vs_predicted(pred_df)
    plot_feature_importance(model, FEATURE_COLS)

    print("\n=== Training complete. ===")
    return metrics


if __name__ == "__main__":
    metrics = run_training()
    print(f"\nFinal metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
