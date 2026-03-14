"""
agent_index.py
---------------
Builds a searchable knowledge base from your data.

How it works:
  1. Reads summary CSV + blackspots CSV + model metrics
  2. Converts each row into a human-readable text document
  3. Converts each document into a vector (list of numbers) using
     a local embedding model (runs on your machine, no API needed)
  4. Stores all vectors in a FAISS index for fast similarity search

When a user asks a question, their question is also converted to
a vector, then we find the most similar documents and pass them
to the AI as context.

Run with:
    python src/agent_index.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    SUMMARY_CSV, BLACKSPOTS_CSV, PREDICTIONS_CSV,
    FAISS_INDEX_FILE, FAISS_META_FILE,
    FAISS_INDEX_DIR, EMB_MODEL, ensure_dirs,
    OUTPUT_DIR,
)


def load_embedding_model():
    """
    Loads the sentence-transformer model that converts text to vectors.
    Downloads automatically on first run (~90MB), then cached locally.
    """
    print(f"  Loading embedding model: {EMB_MODEL}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL)
    print("  Embedding model ready.")
    return model


def build_documents() -> list[dict]:
    """
    Converts all our data files into plain-English text documents.
    Each document becomes one searchable entry in the index.
    Returns a list of dicts, each with 'id', 'text', 'source'.
    """
    docs = []

    # ── 1. National overview document ─────────────────────────
    docs.append({
        "id":     "overview_001",
        "source": "overview",
        "text": (
            "India Road Accident Overview: "
            "The Integrated Road Accident Database (iRAD) is a central accident "
            "database management system developed by NIC and implemented across "
            "all 36 states and union territories of India. "
            "iRAD was launched on January 13, 2020 by the Ministry of Road "
            "Transport and Highways. As of January 2025, iRAD has recorded "
            "over 14,66,962 total accidents involving 34,43,441 persons. "
            "Of these, 5,43,954 were killed and 6,61,849 suffered grievous injuries. "
            "The eDAR (e-Detailed Accident Report) went live in 2024, enabling "
            "cashless treatment for road accident victims."
        ),
    })

    # ── 2. Campbell intervention evidence documents ────────────
    interventions = [
        {
            "id": "campbell_engineering",
            "source": "campbell_review",
            "text": (
                "Road Engineering Interventions (Campbell Review Section 7): "
                "Engineering measures have strong evidence for reducing accidents. "
                "Key interventions include: installing crash barriers and guardrails "
                "at high-risk locations, improving road geometry at dangerous junctions, "
                "adding rumble strips on rural highways, improving road lighting "
                "especially at night accident hotspots, installing speed bumps near "
                "schools and pedestrian crossings, and constructing grade-separated "
                "crossings on national highways. Evidence shows engineering measures "
                "can reduce fatal accidents by 20-40% at treated locations."
            ),
        },
        {
            "id": "campbell_enforcement",
            "source": "campbell_review",
            "text": (
                "Road Enforcement Interventions (Campbell Review Section 7): "
                "Enforcement measures show consistent evidence of effectiveness. "
                "Speed cameras reduce speeding by 10-15% and fatal accidents by "
                "20-30% at treated sites. Drunk driving checkpoints reduce "
                "alcohol-related crashes by 15-20%. Automated enforcement systems "
                "deployed on national highways in India have shown measurable "
                "reductions in over-speeding violations. Helmet and seatbelt "
                "enforcement campaigns reduce head injury fatalities significantly. "
                "The iRAD system supports enforcement by identifying high-violation corridors."
            ),
        },
        {
            "id": "campbell_education",
            "source": "campbell_review",
            "text": (
                "Road Safety Education Interventions (Campbell Review Section 8): "
                "Education campaigns targeting specific behaviours show moderate evidence. "
                "School-based road safety programs reduce child pedestrian casualties. "
                "Two-wheeler safety campaigns are critical in India where two-wheelers "
                "account for 38-52% of all accidents. Driver training programs "
                "and graduated licensing systems for young drivers reduce novice "
                "driver crashes by 15-25%. Community-based campaigns combined with "
                "enforcement show stronger effects than standalone education. "
                "The iSAFE program and Indian Road Traffic Education (IRTE) are "
                "examples of education initiatives under the iRAD 4E framework."
            ),
        },
        {
            "id": "campbell_emergency",
            "source": "campbell_review",
            "text": (
                "Emergency Care Interventions (Campbell Review Section 8): "
                "Post-crash emergency response critically affects fatality rates. "
                "Reducing ambulance response time from 30 to 15 minutes can "
                "prevent 15-20% of road accident deaths. Delhi iRAD has integrated "
                "GPS tracking of 261 CATS ambulances for dynamic repositioning "
                "to accident-prone areas. The eDAR system enables cashless "
                "treatment authorisation for victims within minutes of a crash. "
                "Trauma centre availability within 30 minutes of black spots "
                "is a key recommendation. Good Samaritan protections encourage "
                "bystander assistance and are supported by the iRAD framework."
            ),
        },
    ]
    docs.extend(interventions)

    # ── 3. State summary documents ─────────────────────────────
    if SUMMARY_CSV.exists():
        summary = pd.read_csv(SUMMARY_CSV)
        for _, row in summary.iterrows():
            docs.append({
                "id":     f"state_{row['state'].replace(' ', '_').lower()}",
                "source": "state_summary",
                "text": (
                    f"State Summary for {row['state']}: "
                    f"Total accidents recorded: {int(row['total_accidents']):,}. "
                    f"Total persons killed: {int(row['total_killed']):,}. "
                    f"Total grievous injuries: {int(row.get('grievous_injury', row.get('total_grievous', 0))):,}. "
                    f"Average fatality rate: {row['avg_fatality_rate']:.2f} deaths "
                    f"per 100 accidents. "
                    f"Average severity index: {row.get('avg_severity', row.get('fatality_rate', 0)):.4f}. "
                    f"Data available for {int(row.get('years_recorded', row.get('years_of_data', 5)))} years."
                ),
            })
        print(f"  Added {len(summary)} state summary documents.")

    # ── 4. Black spot documents ────────────────────────────────
    if BLACKSPOTS_CSV.exists():
        bs = pd.read_csv(BLACKSPOTS_CSV)
        critical = bs[bs["risk_level"].isin(["Critical", "High"])]
        for _, row in critical.iterrows():
            docs.append({
                "id":     f"blackspot_{row['state'].replace(' ', '_').lower()}",
                "source": "blackspot",
                "text": (
                    f"Black Spot Alert for {row['state']} ({int(row['year'])}): "
                    f"Risk level: {row['risk_level']}. "
                    f"Total accidents: {int(row['total_accidents']):,}. "
                    f"Deaths: {int(row['killed']):,}. "
                    f"Fatality rate: {row['fatality_rate']:.2f}%. "
                    f"Recommended interventions: {row['recommended_interventions']}."
                ),
            })
        print(f"  Added {len(critical)} black spot documents.")

    # ── 5. Model metrics document ──────────────────────────────
    metrics_path = OUTPUT_DIR / "model_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path).iloc[0]
        docs.append({
            "id":     "model_metrics_001",
            "source": "model",
            "text": (
                f"Poisson Regression Model Results: "
                f"The model predicts road accident counts using features: "
                f"{metrics['features_used']}. "
                f"Mean Absolute Error on test set: {metrics['mae']:,.0f} accidents. "
                f"Root Mean Square Error: {metrics['rmse']:,.0f} accidents. "
                f"Mean Absolute Percentage Error: {metrics['mape_pct']:.1f}%. "
                f"Training set size: {int(metrics['train_rows'])} rows. "
                f"Poisson regression is appropriate for count data and is used "
                f"in official road safety research worldwide."
            ),
        })
        print(f"  Added model metrics document.")

    print(f"  Total documents built: {len(docs)}")
    return docs


def embed_documents(docs: list[dict], emb_model) -> np.ndarray:
    """
    Converts all document texts into vectors using the embedding model.
    Returns a 2D numpy array of shape (num_docs, embedding_dim).
    """
    texts = [d["text"] for d in docs]
    print(f"  Embedding {len(texts)} documents (this may take 1-2 minutes)...")
    embeddings = emb_model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,   # normalise so cosine similarity = dot product
    )
    return np.array(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray):
    """
    Creates a FAISS index from the embeddings.
    IndexFlatIP = exact inner product search (works perfectly with normalised vectors).
    """
    import faiss
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  FAISS index built: {index.ntotal} vectors, {dim} dimensions.")
    return index


def run_index_build():
    ensure_dirs()
    print("\n=== Building Agent Knowledge Index ===\n")

    # Load embedding model
    emb_model = load_embedding_model()

    # Build text documents from data
    print("\nBuilding documents from data files...")
    docs = build_documents()

    # Embed documents
    print("\nEmbedding documents...")
    embeddings = embed_documents(docs, emb_model)

    # Build FAISS index
    print("\nBuilding FAISS index...")
    index = build_faiss_index(embeddings)

    # Save FAISS index
    import faiss
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    print(f"  FAISS index saved  ->  {FAISS_INDEX_FILE}")

    # Save metadata (doc texts + ids) alongside the index
    meta = {
        "meta":  [{"id": d["id"], "source": d["source"]} for d in docs],
        "texts": [d["text"] for d in docs],
    }
    with open(FAISS_META_FILE, "wb") as f:
        pickle.dump(meta, f)
    print(f"  Metadata saved     ->  {FAISS_META_FILE}")

    print(f"\n=== Index build complete. {len(docs)} documents indexed. ===")
    return len(docs)


if __name__ == "__main__":
    run_index_build()
