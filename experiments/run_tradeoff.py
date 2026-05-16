"""
Privacy-Utility Trade-off Experiment
=====================================
Compares two DP mechanisms for private similarity search in ChromaDB:

  1. Gaussian Mechanism  — (ε, δ)-DP noise injected into stored embeddings
     Standard approach; σ = sensitivity × √(2 ln(1.25/δ)) / ε

  2. Sparse Vector Technique (SVT)  — ε-DP query-answering
     Lyu, Su & Li, "Understanding the Sparse Vector Technique for
     Differential Privacy", VLDB 2017. https://arxiv.org/abs/1603.01699

     SVT pays a FIXED total ε regardless of how many documents are checked,
     by only releasing the identity of documents above a similarity threshold.

Results saved to results/tradeoff.csv
"""

import sys
import csv
import numpy as np

sys.path.insert(0, ".")

from src.dp import (
    add_dp_noise, compute_sigma, recall_at_k,
    svt_retrieve,
)
from src.embeddings import embed
from src.store import get_client, upsert, query

DOCUMENTS = [
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
    "Blood pressure reading: 138/88 mmHg — borderline hypertensive.",
    "Retirement savings total $280,000 across IRA and 401k accounts.",
    "Patient is allergic to penicillin and sulfa-based antibiotics.",
]
DOC_IDS = [f"doc_{i}" for i in range(len(DOCUMENTS))]
QUERY_TEXT = "Patient has high blood pressure and diabetes medication."
EPSILONS = [0.1, 0.5, 1.0, 5.0, 10.0]
DELTA = 1e-5
REPEATS = 5          # trials per ε for stable recall estimates
SVT_THRESHOLD = 0.3  # cosine similarity threshold for SVT
SVT_C = 3            # max above-threshold results to return


def gaussian_recall(
    raw: np.ndarray,
    query_emb: np.ndarray,
    gt_ids: list[str],
    client,
    epsilon: float,
) -> float:
    """Average Recall@3 for Gaussian mechanism over REPEATS trials."""
    col_name = f"gauss_{str(epsilon).replace('.', '_')}"
    col = client.get_or_create_collection(col_name, metadata={"hnsw:space": "cosine"})
    recalls = []
    for _ in range(REPEATS):
        noisy_embs = np.array([add_dp_noise(e.copy(), epsilon) for e in raw])
        upsert(col, DOC_IDS, noisy_embs, DOCUMENTS, {"epsilon_used": str(epsilon)})
        noisy_q = add_dp_noise(query_emb.copy(), epsilon)
        res = query(col, noisy_q)
        recalls.append(recall_at_k(res["ids"][0], gt_ids))
    client.delete_collection(col_name)
    return float(np.mean(recalls))


def svt_recall(
    raw: np.ndarray,
    query_emb: np.ndarray,
    gt_ids: list[str],
    epsilon: float,
) -> float:
    """
    Average Recall@3 for SVT (Lyu et al. 2017) over REPEATS trials.

    SVT checks each document's cosine similarity against the query and
    returns the first SVT_C documents above SVT_THRESHOLD — paying a
    fixed ε total, not ε per document.
    """
    recalls = []
    for _ in range(REPEATS):
        retrieved = svt_retrieve(
            query_emb, raw, DOC_IDS,
            threshold=SVT_THRESHOLD,
            epsilon=epsilon,
            c=SVT_C,
        )
        recalls.append(recall_at_k(retrieved, gt_ids))
    return float(np.mean(recalls))


def run() -> None:
    np.random.seed(42)

    print("=" * 65)
    print("Differential Privacy × ChromaDB — Trade-off Experiment")
    print("=" * 65)

    # ── Embeddings ────────────────────────────────────────────────────────
    print("\n[1/4] Generating embeddings...")
    raw = embed(DOCUMENTS)
    query_emb = embed([QUERY_TEXT])[0]
    print(f"      Shape: {raw.shape}")

    # ── ChromaDB ──────────────────────────────────────────────────────────
    print("[2/4] Connecting to ChromaDB...")
    client = get_client()

    # ── Ground truth (raw, no noise) ──────────────────────────────────────
    print("[3/4] Building ground-truth collection (no noise)...")
    raw_col = client.get_or_create_collection(
        "raw_embeddings", metadata={"hnsw:space": "cosine"}
    )
    upsert(raw_col, DOC_IDS, raw, DOCUMENTS, {"epsilon_used": "none"})
    gt_res = query(raw_col, query_emb)
    gt_ids = gt_res["ids"][0]

    print("      Ground-truth top-3:")
    for doc, dist in zip(gt_res["documents"][0], gt_res["distances"][0]):
        print(f"        cos_dist={dist:.4f}  {doc[:60]}")

    # ── Epsilon sweep ─────────────────────────────────────────────────────
    print("\n[4/4] Running epsilon sweep (Gaussian vs SVT)...")
    print(f"\n{'ε':>6} | {'σ (Gauss)':>10} | {'Recall@3 Gaussian':>18} | {'Recall@3 SVT':>13}")
    print("-" * 58)

    rows = []
    for eps in EPSILONS:
        sigma = compute_sigma(eps, DELTA, sensitivity=1.0)
        g_rec = gaussian_recall(raw, query_emb, gt_ids, client, eps)
        s_rec = svt_recall(raw, query_emb, gt_ids, eps)
        print(f"{eps:>6.1f} | {sigma:>10.4f} | {g_rec:>18.2f} | {s_rec:>13.2f}")
        rows.append({
            "epsilon": eps,
            "sigma": round(sigma, 4),
            "recall_gaussian": round(g_rec, 2),
            "recall_svt": round(s_rec, 2),
        })

    # ── Save CSV ──────────────────────────────────────────────────────────
    out = "results/tradeoff.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epsilon", "sigma", "recall_gaussian", "recall_svt"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {out}")

    # ── Conclusion ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CONCLUSION")
    print("=" * 65)
    print(f"""
Gaussian Mechanism (ε, δ)-DP:
  Injects calibrated noise into every stored embedding.
  σ scales as 1/ε — at low ε the noise overwhelms the 384-dim signal.

Sparse Vector Technique — Lyu, Su & Li (VLDB 2017):
  Answers a stream of similarity threshold queries with a FIXED ε cost,
  regardless of how many documents are checked.
  AboveThreshold adds Lap(2/ε) to the threshold and Lap(4/ε) per query.
  Sparse runs AboveThreshold up to c={SVT_C} times, splitting ε/c each time.

Key difference:
  Gaussian pays ε per stored embedding (storage-side noise).
  SVT pays a fixed ε for the entire query stream (query-side privacy).
  They are complementary: combine both for end-to-end protection.

Averaged over {REPEATS} trials per ε. Full results in results/tradeoff.csv.
""")


if __name__ == "__main__":
    run()
