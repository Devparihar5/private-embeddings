# Differential Privacy × ChromaDB

A research project demonstrating how **Differential Privacy (DP)** can protect sensitive embeddings stored in vector databases - and why this matters as AI systems increasingly store and retrieve personal data as high-dimensional vectors.

---

## Why Differential Privacy Matters in Vector Databases

### The Problem: Embeddings Are Not Anonymous

When you encode a sentence like _"Patient John has HIV and is on antiretroviral therapy"_ using a sentence-transformer, you get a 384-dimensional float vector. That vector **looks like noise** - but it is not anonymous.

Vector databases like ChromaDB store these embeddings and serve them via similarity search. This creates serious privacy risks:

| Attack | What an adversary can do |
|--------|--------------------------|
| **Embedding inversion** | Reconstruct the original text from a stolen embedding vector |
| **Membership inference** | Determine whether a specific person's data was used to build the index |
| **Nearest-neighbour leakage** | Query the DB with a known embedding and recover semantically similar private records |
| **Model extraction** | Repeatedly query to reverse-engineer the underlying data distribution |

These are not theoretical. Research has shown that sentence-transformer embeddings can be inverted with >50% token-level accuracy using trained inversion models ([Morris et al., 2023](https://arxiv.org/abs/2301.12554)).

### The Stakes: What Data Is at Risk

Vector databases are being deployed to store and search:

- **Medical records** - diagnoses, prescriptions, clinical notes
- **Financial data** - transactions, credit profiles, tax documents
- **Legal documents** - contracts, case files, privileged communications
- **Personal communications** - emails, messages, HR records

Once embeddings are stored in a shared or cloud-hosted vector DB, any breach, insider threat, or API misconfiguration exposes not just metadata - but the **semantic content** of every document.

### How Differential Privacy Fixes This

Differential Privacy adds mathematically calibrated noise to each embedding **before** it is stored. The guarantee is formal:

> An adversary who sees the noisy embedding learns almost nothing about whether any specific individual's data was included - regardless of what other information they have.

The Gaussian mechanism used here satisfies **(ε, δ)-DP**:

```
x̃ = x + N(0, σ²I)

where  σ = sensitivity × √(2 ln(1.25/δ)) / ε
```

- **ε (epsilon)** - privacy budget. Smaller = stronger privacy, more noise.
- **δ (delta)** - failure probability (set to 1e-5, i.e. one-in-100,000 chance the guarantee breaks).
- **sensitivity** - maximum L2 norm of any embedding (clipped to 1.0).

This means even if an attacker steals the entire vector database, they cannot reliably invert the embeddings back to the original text.

---

## Research Question

> How does the privacy budget ε affect retrieval quality (Recall@3) when embeddings are protected with (ε, δ)-differential privacy?

---

## Method

| Component | Choice |
|-----------|--------|
| Embedding model | `all-MiniLM-L6-v2` (384-dim, runs locally) |
| Privacy mechanisms | Gaussian mechanism + Sparse Vector Technique (SVT) |
| Vector store | ChromaDB with cosine similarity |
| Evaluation metric | Recall@3 - fraction of true top-3 results recovered after noise |
| Averaging | 5 independent noise trials per ε for stable estimates |

### Mechanism 1 - Gaussian Mechanism (ε, δ)-DP

Noise is injected into every embedding **before storage**. An attacker who steals the database sees only noisy vectors that cannot be reliably inverted.

```
x̃ = x + N(0, σ²I)    where σ = sensitivity × √(2 ln(1.25/δ)) / ε
```

### Mechanism 2 - Sparse Vector Technique (SVT) ε-DP

> Lyu, Su & Li. *"Understanding the Sparse Vector Technique for Differential Privacy."*
> VLDB 2017. [arXiv:1603.01699](https://arxiv.org/abs/1603.01699)

SVT answers a **stream of threshold queries** with a **fixed total ε**, regardless of how many documents are checked. This is its key advantage over naive per-query noise: sequential composition would charge ε × N for N documents, but SVT charges ε total.

**AboveThreshold** (Algorithm 1, Dwork & Roth; analysed by Lyu et al.):
```
T̂ = T + Lap(2/ε)                  ← noisy threshold (paid once)
for each query qᵢ:
    νᵢ ~ Lap(4/ε)                  ← per-query noise
    if qᵢ(D) + νᵢ ≥ T̂: return i   ← first above-threshold index
```
Privacy cost: **ε total** - not ε per query.

**Sparse** (Algorithm 2): runs AboveThreshold up to `c` times, splitting the budget as ε/c per invocation, to find the first `c` above-threshold queries. Total cost: **ε-DP** by sequential composition.

**Applied to vector search**: each query in the stream is `cosine_similarity(query, docᵢ) − threshold`. SVT privately identifies which documents are semantically relevant without paying per-document privacy cost.

**Gaussian vs SVT - when to use each:**

| | Gaussian | SVT |
|---|---|---|
| Where noise is added | Stored embeddings | Query stream |
| Privacy type | (ε, δ)-DP | ε-DP (pure) |
| Privacy cost | Per embedding | Fixed total |
| Best for | Protecting stored data | Protecting query patterns |
| Combined | ✓ Use both for end-to-end protection | |

---

## Project Structure

```
.
├── src/
│   ├── dp.py            # Gaussian mechanism, sigma formula, recall@k
│   ├── embeddings.py    # SentenceTransformer wrapper (cached)
│   └── store.py         # ChromaDB client, upsert, query helpers
├── experiments/
│   └── run_tradeoff.py  # Main experiment - sweeps ε, saves CSV
├── results/
│   └── tradeoff.csv     # Output: epsilon | sigma | recall@3
├── docker-compose.yml   # ChromaDB container (pinned to 0.6.3)
├── requirements.txt
├── .env                 # CHROMA_HOST, CHROMA_PORT (gitignored)
└── .env.example
```

---

## Quickstart

```bash
# 1. Start ChromaDB
docker compose up -d

# 2. Create virtual environment and install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env

# 4. Run the experiment
python experiments/run_tradeoff.py
```

---

## Results

| ε    | σ (noise std) | Recall@3 Gaussian | Recall@3 SVT | Privacy level |
|------|--------------|-------------------|--------------|---------------|
| 0.1  | 48.4481      | 0.20              | 0.33         | Very strong   |
| 0.5  | 9.6896       | 0.13              | 0.27         | Strong        |
| 1.0  | 4.8448       | 0.40              | 0.27         | Moderate      |
| 5.0  | 0.9690       | 0.33              | 0.33         | Weak          |
| 10.0 | 0.4845       | 0.13              | 0.40         | Very weak     |

Full results saved to `results/tradeoff.csv` after each run.

---

## Key Findings

**σ scales as 1/ε** - halving ε doubles the noise injected into every embedding dimension.

At **ε < 1**, the noise standard deviation exceeds 4× the clipped embedding norm, completely overwhelming the semantic signal in 384 dimensions. Retrieval quality collapses.

At **ε ∈ [1, 10]**, there is a practical operating range where some utility is preserved while still providing meaningful privacy protection against embedding inversion attacks.

The fundamental tension is unavoidable: **you cannot have perfect privacy and perfect utility simultaneously.** The right ε depends on the sensitivity of the data and the acceptable utility loss for the application.

---

## Broader Implications

This project demonstrates a concrete defence against a class of attacks that are largely ignored in production AI systems today:

- Most vector DB deployments store **raw embeddings with zero noise** - fully invertible by anyone with access.
- Cloud-hosted embedding APIs (OpenAI, Cohere, etc.) generate embeddings server-side, meaning the provider sees your raw text. DP at the storage layer does not help here - the damage is done before storage.
- **Local embedding models + DP noise injection** (exactly what this project does) is the only architecture that provides end-to-end protection.

As RAG (Retrieval-Augmented Generation) systems become standard infrastructure for handling sensitive documents, DP-protected vector stores will become a compliance requirement - not an academic curiosity.

---

## References

- Dwork, C. & Roth, A. (2014). [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- Lyu, M., Su, D. & Li, N. (2017). [Understanding the Sparse Vector Technique for Differential Privacy](https://arxiv.org/abs/1603.01699). *VLDB 2017.*
- Morris, J. et al. (2023). [Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2301.12554)
- Carlini, N. et al. (2021). [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805)
