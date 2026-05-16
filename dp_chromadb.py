"""
Differential Privacy applied to ChromaDB vector database.
Demonstrates privacy-utility trade-off using Gaussian mechanism noise injection.
"""

import math
import os
import numpy as np
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List

load_dotenv()

DOCUMENTS: List[str] = [
    "Patient has a history of hypertension and type 2 diabetes.",
    "Annual salary is $95,000 with a 401k contribution of 6%.",
    "MRI scan revealed a small lesion in the left temporal lobe.",
    "Credit score is 720 with outstanding mortgage balance of $340,000.",
    "Prescribed metformin 500mg twice daily for blood sugar control.",
    "Investment portfolio contains 60% equities and 40% bonds.",
    "Patient reports chronic lower back pain for the past three years.",
    "Tax return shows adjusted gross income of $112,000 last fiscal year.",
    "Colonoscopy results indicate no polyps; next screening in 10 years.",
    "Checking account balance is $4,200 with two pending transactions.",
    "Diagnosed with generalized anxiety disorder; referred to therapist.",
    "Home equity line of credit approved for $50,000 at 7.5% APR.",
    "Blood pressure reading: 138/88 mmHg - borderline hypertensive.",
    "Retirement savings total $280,000 across IRA and 401k accounts.",
    "Patient is allergic to penicillin and sulfa-based antibiotics.",
]
DOC_IDS: List[str] = [f"doc_{i}" for i in range(len(DOCUMENTS))]
QUERY_TEXT = "Patient has high blood pressure and diabetes medication."


# ---------------------------------------------------------------------------
# Differential Privacy
# ---------------------------------------------------------------------------

def compute_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    """Compute Gaussian mechanism sigma: σ = sensitivity * sqrt(2*ln(1.25/δ)) / ε."""
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def add_dp_noise(
    embedding: np.ndarray,
    epsilon: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
) -> np.ndarray:
    """Clip embedding to L2 norm ≤ sensitivity, then add calibrated Gaussian noise."""
    norm = np.linalg.norm(embedding)
    if norm > sensitivity:
        embedding = embedding * (sensitivity / norm)
    sigma = compute_sigma(epsilon, delta, sensitivity)
    return embedding + np.random.normal(0, sigma, size=embedding.shape)


def apply_epsilon_privacy(
    client: chromadb.ClientAPI,
    raw_embeddings: np.ndarray,
    query_emb: np.ndarray,
    gt_ids: List[str],
    epsilon: float,
    delta: float = 1e-5,
) -> tuple[float, float]:
    """
    Build a temporary DP collection for a given epsilon, query it,
    and return (sigma, recall@3) against the ground-truth top-3 IDs.
    """
    sigma = compute_sigma(epsilon, delta, sensitivity=1.0)
    col_name = f"tmp_eps_{str(epsilon).replace('.', '_')}"

    col = client.get_or_create_collection(col_name, metadata={"hnsw:space": "cosine"})
    noisy_embs = np.array([add_dp_noise(e.copy(), epsilon) for e in raw_embeddings])
    col.upsert(
        ids=DOC_IDS,
        embeddings=noisy_embs.tolist(),
        documents=DOCUMENTS,
        metadatas=[{"epsilon_used": str(epsilon)} for _ in DOC_IDS],
    )

    noisy_q = add_dp_noise(query_emb.copy(), epsilon)
    results = col.query(query_embeddings=[noisy_q.tolist()], n_results=3,
                        include=["documents", "distances"])
    recall = len(set(results["ids"][0]) & set(gt_ids)) / len(gt_ids)

    client.delete_collection(col_name)
    return sigma, recall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Return (N, D) float32 embeddings."""
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def upsert_collection(
    col: chromadb.Collection,
    embeddings: np.ndarray,
    extra_meta: dict,
) -> None:
    """Upsert all documents into a collection."""
    col.upsert(
        ids=DOC_IDS,
        embeddings=embeddings.tolist(),
        documents=DOCUMENTS,
        metadatas=[{**extra_meta, "doc_id": d} for d in DOC_IDS],
    )


def query_collection(col: chromadb.Collection, emb: np.ndarray, n: int = 3):
    """Query collection and return result."""
    return col.query(query_embeddings=[emb.tolist()], n_results=n,
                     include=["documents", "distances"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)

    # ── 1. Model ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading sentence-transformer model")
    print("=" * 60)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Loaded: all-MiniLM-L6-v2")

    # ── 2. Embeddings ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Generating raw embeddings")
    print("=" * 60)
    raw_embeddings = embed_texts(model, DOCUMENTS)
    print(f"Shape : {raw_embeddings.shape}")
    print(f"Sample: {raw_embeddings[0][:6]} ...")

    # ── 3. ChromaDB ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 3: Connecting to ChromaDB ({os.getenv('CHROMA_HOST','localhost')}:{os.getenv('CHROMA_PORT',8000)})")
    print("=" * 60)
    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", 8000)),
    )

    raw_col = client.get_or_create_collection("raw_embeddings", metadata={"hnsw:space": "cosine"})
    dp_col  = client.get_or_create_collection("dp_embeddings",  metadata={"hnsw:space": "cosine"})

    upsert_collection(raw_col, raw_embeddings, {"epsilon_used": "none"})
    print("Stored raw embeddings.")

    dp_eps = 1.0
    dp_embs = np.array([add_dp_noise(e.copy(), dp_eps) for e in raw_embeddings])
    upsert_collection(dp_col, dp_embs, {"epsilon_used": str(dp_eps)})
    print(f"Stored DP embeddings (ε={dp_eps}).")

    # ── 4. Query comparison ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f'STEP 4: Query - "{QUERY_TEXT}"')
    print("=" * 60)
    query_emb = embed_texts(model, [QUERY_TEXT])[0]

    raw_res = query_collection(raw_col, query_emb)
    dp_res  = query_collection(dp_col, add_dp_noise(query_emb.copy(), dp_eps))

    print("\n--- Raw top-3 ---")
    for doc, dist in zip(raw_res["documents"][0], raw_res["distances"][0]):
        print(f"  {dist:.4f}  |  {doc[:70]}")

    print(f"\n--- DP top-3 (ε={dp_eps}) ---")
    for doc, dist in zip(dp_res["documents"][0], dp_res["distances"][0]):
        print(f"  {dist:.4f}  |  {doc[:70]}")

    gt_ids = raw_res["ids"][0]
    recall = len(set(dp_res["ids"][0]) & set(gt_ids)) / len(gt_ids)
    print(f"\nRecall@3 (ε={dp_eps}): {recall:.2f}")

    # ── 5. Privacy-utility trade-off ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Privacy-Utility Trade-off Analysis")
    print("=" * 60)
    print(f"\n{'epsilon':>10} | {'sigma':>10} | {'Recall@3':>10}")
    print("-" * 38)

    for eps in [0.1, 0.5, 1.0, 5.0, 10.0]:
        sigma, rec = apply_epsilon_privacy(client, raw_embeddings, query_emb, gt_ids, eps)
        print(f"{eps:>10.1f} | {sigma:>10.4f} | {rec:>10.2f}")

    # ── 6. Conclusion ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
Lower ε → higher σ → more noise → lower Recall@3  (strong privacy, poor utility)
Higher ε → lower σ → less noise → higher Recall@3 (weak privacy,  good utility)

ε ∈ [1, 10] is the practical operating range for most applications.
""")


if __name__ == "__main__":
    main()
