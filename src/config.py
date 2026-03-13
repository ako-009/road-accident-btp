"""
config.py
---------
Central settings file. Every other module imports from here.
Change a setting once here and it updates everywhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(override=False)

# Project root = the folder that contains src/, data/, outputs/
ROOT_DIR = Path(__file__).parent.parent

# ── Data paths ────────────────────────────────────────────────
RAW_DIR       = ROOT_DIR / os.getenv("RAW_DIR",       "data/raw")
PROCESSED_DIR = ROOT_DIR / os.getenv("PROCESSED_DIR", "data/processed")
CLEAN_CSV     = PROCESSED_DIR / "accidents_cleaned.csv"

# ── Output paths ──────────────────────────────────────────────
OUTPUT_DIR      = ROOT_DIR / os.getenv("OUTPUT_DIR",      "outputs")
PLOTS_DIR       = ROOT_DIR / os.getenv("PLOTS_DIR",       "outputs/plots")
FAISS_INDEX_DIR = ROOT_DIR / os.getenv("FAISS_INDEX_DIR", "outputs/agent_index")
MODELS_DIR      = ROOT_DIR / os.getenv("MODELS_DIR",      "models")

# Specific output files used by multiple modules
SUMMARY_CSV      = OUTPUT_DIR / "summary_by_state.csv"
PREDICTIONS_CSV  = OUTPUT_DIR / "poisson_predictions.csv"
BLACKSPOTS_CSV   = OUTPUT_DIR / "blackspots.csv"
DATA_DICT_CSV    = OUTPUT_DIR / "data_dictionary.csv"

# FAISS vector search index files
FAISS_INDEX_FILE   = FAISS_INDEX_DIR / "faiss.index"
FAISS_META_FILE    = FAISS_INDEX_DIR / "meta.pkl"

# Saved model file
POISSON_MODEL_FILE = MODELS_DIR / "poisson.pkl"

# ── OpenAI settings ───────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL",   "gpt-4o-mini")

# ── Local embedding model (runs on your machine, free) ────────
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
TOP_K     = int(os.getenv("TOP_K", "5"))

# ── API server settings ───────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


def ensure_dirs():
    """Creates all needed folders if they do not exist yet."""
    for folder in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR,
                   PLOTS_DIR, FAISS_INDEX_DIR, MODELS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("=== Road Accident BTP - Config Check ===")
    print(f"  Project root   : {ROOT_DIR}")
    print(f"  Raw data       : {RAW_DIR}")
    print(f"  Processed data : {PROCESSED_DIR}")
    print(f"  Plots output   : {PLOTS_DIR}")
    print(f"  OpenAI model   : {OPENAI_MODEL}")
    print(f"  Embedding      : {EMB_MODEL}")
    print(f"  API port       : {API_PORT}")
    print("  All folders created successfully!")
