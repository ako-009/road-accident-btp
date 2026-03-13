"""
agent_api.py
-------------
FastAPI backend for the Road Accident AI Agent.

Endpoints:
  POST /chat         - Ask a question, get an AI answer with sources
  GET  /totals       - National accident totals
  GET  /blackspots   - List of black spot states
  GET  /plots        - List of available plot files
  GET  /top-states   - Top states by a chosen metric
  GET  /state/{name} - Summary for one state
  GET  /trend        - Year-over-year national trend
  GET  /health       - Health check

Start the server with:
    uvicorn src.agent_api:app --reload --port 8000
"""

import sys
import pickle
import traceback
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    FAISS_INDEX_FILE, FAISS_META_FILE,
    EMB_MODEL, TOP_K, OPENAI_API_KEY, OPENAI_MODEL,
)
from src.tools import (
    get_national_totals, get_top_states, get_state_summary,
    get_blackspots, get_plot_list, get_model_metrics, get_yearly_trend,
)


# ── FastAPI app setup ─────────────────────────────────────────────
app = FastAPI(
    title="Road Accident AI Agent",
    description="AI-powered road accident analysis for India using iRAD data",
    version="1.0.0",
)

# Allow Streamlit (running on port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global objects (loaded once when server starts) ───────────────
_faiss_index  = None
_faiss_meta   = None
_emb_model    = None


def get_faiss_index():
    """Loads FAISS index from disk (only once, then cached)."""
    global _faiss_index, _faiss_meta
    if _faiss_index is None:
        if not FAISS_INDEX_FILE.exists():
            raise RuntimeError(
                "FAISS index not found. Run python src/agent_index.py first."
            )
        import faiss
        _faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
        with open(FAISS_META_FILE, "rb") as f:
            _faiss_meta = pickle.load(f)
    return _faiss_index, _faiss_meta


def get_emb_model():
    """Loads the sentence-transformer model (only once, then cached)."""
    global _emb_model
    if _emb_model is None:
        from sentence_transformers import SentenceTransformer
        _emb_model = SentenceTransformer(EMB_MODEL)
    return _emb_model


# ── Retrieval helper ──────────────────────────────────────────────
def retrieve_docs(question: str, k: int = TOP_K) -> list[dict]:
    """
    Finds the k most relevant documents for the question.
    Returns a list of dicts with 'text', 'id', 'source', 'score'.
    """
    index, meta = get_faiss_index()
    emb_model   = get_emb_model()

    # Embed the question using the same model used for documents
    q_vec = emb_model.encode(
        [question],
        normalize_embeddings=True,
    ).astype(np.float32)

    # Search for k nearest neighbours
    scores, indices = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append({
            "text":   meta["texts"][idx],
            "id":     meta["meta"][idx]["id"],
            "source": meta["meta"][idx]["source"],
            "score":  round(float(score), 4),
        })
    return results


# ── OpenAI call ───────────────────────────────────────────────────
def call_openai(system_prompt: str, user_message: str) -> str:
    """
    Calls the OpenAI API and returns the assistant's reply.
    Falls back gracefully if the API key is missing.
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-put-your-real-key-here":
        return (
            "OpenAI API key not configured. "
            "Please set OPENAI_API_KEY in your .env file. "
            "Retrieved context is shown in the sources below."
        )
    try:
        from openai import OpenAI
        client   = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message},
            ],
            max_tokens=600,
            temperature=0.3,   # lower = more factual, less creative
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API error: {str(e)}"


# ── System prompt for the agent ───────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI assistant for Indian road accident analysis.
You help researchers, policymakers, and students understand road safety data.

Your knowledge is based on:
- iRAD (Integrated Road Accident Database) — India's official accident database
- Campbell Systematic Review of Road Safety Interventions
- Poisson regression analysis of accident patterns

Rules:
1. Answer ONLY based on the retrieved context documents provided
2. If the context does not contain the answer, say so clearly
3. Always cite specific numbers when available
4. Recommend interventions from the 4E framework: Engineering, Enforcement, Education, Emergency
5. Be concise but thorough — aim for 3-5 sentences
6. If asked about a specific state, focus on that state's data
"""


# ── Request/Response models ───────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    use_tools: Optional[list[str]] = []


class ChatResponse(BaseModel):
    answer:   str
    sources:  list[dict]
    question: str


# ── API Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Simple endpoint to verify the API is running."""
    return {"status": "ok", "message": "Road Accident AI Agent is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Retrieves relevant documents, then asks OpenAI to answer based on them.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Step 1: Retrieve relevant documents
        docs = retrieve_docs(request.question, k=TOP_K)

        # Step 2: Build context string from retrieved docs
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"[Document {i} | source: {doc['source']} | id: {doc['id']}]\n"
                f"{doc['text']}"
            )
        context = "\n\n".join(context_parts)

        # Step 3: Build the user message with context
        user_message = (
            f"Context documents:\n\n{context}\n\n"
            f"User question: {request.question}\n\n"
            f"Answer based on the context above:"
        )

        # Step 4: Call OpenAI
        answer = call_openai(SYSTEM_PROMPT, user_message)

        return ChatResponse(
            answer=answer,
            sources=docs,
            question=request.question,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/totals")
def totals():
    """Returns national accident totals."""
    return get_national_totals()


@app.get("/blackspots")
def blackspots(risk_level: str = "all"):
    """Returns black spot states. Query param: risk_level=Critical|High|all"""
    return get_blackspots(risk_level)


@app.get("/plots")
def plots():
    """Returns list of available plot filenames."""
    return get_plot_list()


@app.get("/top-states")
def top_states(by: str = "killed", year: int = 2022, n: int = 10):
    """Returns top N states. Query params: by=killed|total_accidents|fatality_rate"""
    return get_top_states(by=by, year=year, n=n)


@app.get("/state/{state_name}")
def state_detail(state_name: str):
    """Returns detailed summary for one state."""
    return get_state_summary(state_name)


@app.get("/trend")
def trend():
    """Returns year-over-year national trend data."""
    return get_yearly_trend()


@app.get("/metrics")
def model_metrics():
    """Returns Poisson model performance metrics."""
    return get_model_metrics()
