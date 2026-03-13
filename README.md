# Road Accident Analysis — BTP Project

AI-powered road accident analysis for India using iRAD data,
Poisson regression modelling, and a retrieval-augmented AI agent.

## Quick Start
```bash
# 1. Activate environment
source venv/Scripts/activate   # Git Bash
# or
venv\Scripts\activate.bat      # Windows CMD

# 2. Run the full pipeline
python src/make_synthetic_data.py
python src/etl.py
python src/eda.py
python src/features.py
python src/train_poisson.py
python src/blackspot.py
python src/agent_index.py

# 3. Start the API (Terminal 1)
uvicorn src.agent_api:app --reload --port 8000

# 4. Start the UI (Terminal 2)
streamlit run src/ui_streamlit.py
```

Open http://localhost:8501 in your browser.

## Project Structure
```
road_ai_assistant/
├── data/
│   ├── raw/                  # Raw CSV files
│   └── processed/            # Cleaned data
├── outputs/
│   ├── plots/                # All 11 generated charts
│   └── agent_index/          # FAISS vector index
├── models/                   # Saved Poisson model
├── src/
│   ├── config.py             # Central settings
│   ├── make_synthetic_data.py
│   ├── etl.py                # Data cleaning
│   ├── eda.py                # 11 plots
│   ├── features.py           # Feature engineering
│   ├── train_poisson.py      # Poisson regression
│   ├── blackspot.py          # Black spot identification
│   ├── agent_index.py        # FAISS knowledge index
│   ├── tools.py              # Agent tool functions
│   ├── agent_api.py          # FastAPI backend
│   ├── canned_prompts.json   # Preset questions
│   └── ui_streamlit.py       # Streamlit dashboard
└── tests/
    └── test_etl.py
```

## Key Features

- **ETL Pipeline**: Cleans and validates raw iRAD-style CSV data
- **EDA**: 11 publication-quality charts saved as PNG
- **Poisson Model**: Predicts accident counts by state/year
- **Black Spot Identification**: Flags high-risk states with interventions
- **AI Agent**: FAISS retrieval + OpenAI for natural language Q&A
- **Dashboard**: 4-tab Streamlit UI with live API integration

## 4E Framework (Campbell Review)

Based on Campbell Systematic Review Sections 7-8:

| E | Intervention | Evidence |
|---|---|---|
| Engineering | Barriers, lighting, road geometry | Strong |
| Enforcement | Speed cameras, DUI checkpoints | Strong |
| Education | Campaigns, driver training | Moderate |
| Emergency | Ambulance response, trauma care | Strong |
