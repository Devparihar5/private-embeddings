"""
Differential Privacy mechanisms.

Implements:
  1. Gaussian mechanism  — (ε, δ)-DP noise injection for embeddings
  2. Sparse Vector Technique (SVT) — AboveThreshold + Sparse
     Based on: Lyu, Su & Li, "Understanding the Sparse Vector Technique
     for Differential Privacy", VLDB 2017.
     https://arxiv.org/abs/1603.01699
"""

import math
import numpy as np
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Gaussian Mechanism  (ε, δ)-DP
# ---------------------------------------------------------------------------

def compute_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    """
    Gaussian mechanism noise scale.
    σ = sensitivity × √(2 ln(1.25/δ)) / ε
    """
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def add_dp_noise(
    embedding: np.ndarray,
    epsilon: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
) -> np.ndarray:
    """
    Clip embedding to L2 norm ≤ sensitivity, then add N(0, σ²I) noise.
    Satisfies (ε, δ)-DP per the Gaussian mechanism.
    """
    norm = np.linalg.norm(embedding)
    if norm > sensitivity:
        embedding = embedding * (sensitivity / norm)
    sigma = compute_sigma(epsilon, delta, sensitivity)
    return embedding + np.random.normal(0, sigma, size=embedding.shape)


# ---------------------------------------------------------------------------
# Sparse Vector Technique  ε-DP
# Lyu, Su & Li — VLDB 2017, §3
# ---------------------------------------------------------------------------

def above_threshold(
    queries: list[Callable[[], float]],
    threshold: float,
    epsilon: float,
) -> Optional[int]:
    """
    AboveThreshold (Algorithm 1, Dwork & Roth; analysed in Lyu et al. 2017).

    Given a stream of sensitivity-1 queries (as zero-argument callables that
    return a float), returns the index of the FIRST query whose noisy answer
    exceeds a noisy threshold.  Satisfies ε-DP regardless of stream length.

    Noise scales (from Lyu et al. §3.1):
      - Threshold noise : Lap(2/ε)
      - Per-query noise : Lap(4/ε)

    Args:
        queries:   List of callables () -> float, each with sensitivity 1.
        threshold: The comparison threshold T.
        epsilon:   Privacy budget for the entire call.

    Returns:
        Index of the first above-threshold query, or None if none found.
    """
    t_hat = threshold + np.random.laplace(loc=0, scale=2.0 / epsilon)
    for idx, q in enumerate(queries):
        nu = np.random.laplace(loc=0, scale=4.0 / epsilon)
        if q() + nu >= t_hat:
            return idx
    return None


def sparse(
    queries: list[Callable[[], float]],
    threshold: float,
    epsilon: float,
    c: int,
) -> list[int]:
    """
    Sparse (Algorithm 2, Dwork & Roth; analysed in Lyu et al. 2017).

    Finds the indices of the first c queries whose answers exceed the
    threshold.  Satisfies ε-DP by splitting the budget across c invocations
    of AboveThreshold (ε_i = ε/c each).

    Args:
        queries:   List of callables () -> float, each with sensitivity 1.
        threshold: The comparison threshold T.
        epsilon:   Total privacy budget.
        c:         Maximum number of above-threshold answers to return.

    Returns:
        List of indices (up to c) of queries that exceeded the threshold.
    """
    idxs: list[int] = []
    pos = 0
    epsilon_i = epsilon / c

    while pos < len(queries) and len(idxs) < c:
        next_idx = above_threshold(queries[pos:], threshold, epsilon_i)
        if next_idx is None:
            break
        abs_idx = pos + next_idx
        idxs.append(abs_idx)
        pos = abs_idx + 1

    return idxs


def svt_retrieve(
    query_emb: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_ids: list[str],
    threshold: float,
    epsilon: float,
    c: int,
) -> list[str]:
    """
    Use SVT to privately identify which documents are semantically relevant.

    Each query in the stream is: cosine_similarity(query_emb, doc_i) - threshold.
    SVT finds the first c documents whose (noisy) similarity exceeds the
    threshold, paying a fixed ε regardless of how many documents are checked.

    Args:
        query_emb:      Query embedding vector (will be L2-normalised).
        doc_embeddings: (N, D) matrix of document embeddings.
        doc_ids:        Corresponding document IDs.
        threshold:      Cosine similarity threshold (e.g. 0.3).
        epsilon:        Total privacy budget for the SVT call.
        c:              Maximum number of relevant documents to return.

    Returns:
        List of doc_ids identified as above-threshold by SVT.
    """
    # Normalise for cosine similarity
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    docs_norm = doc_embeddings / (
        np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-9
    )

    # Build stream of sensitivity-1 queries: sim(q, doc_i) - threshold
    # Cosine similarity ∈ [-1, 1], so sensitivity = 1 (one doc change shifts
    # one query by at most 1).
    queries = [
        (lambda i: lambda: float(np.dot(q, docs_norm[i])))(i)
        for i in range(len(doc_ids))
    ]

    indices = sparse(queries, threshold, epsilon, c)
    return [doc_ids[i] for i in indices]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: list[str]) -> float:
    """Fraction of relevant IDs recovered in retrieved list."""
    return len(set(retrieved) & set(relevant)) / len(relevant) if relevant else 0.0
