"""ChromaDB collection helpers."""

import os
import numpy as np
import chromadb
from dotenv import load_dotenv

load_dotenv()


def get_client() -> chromadb.ClientAPI:
    """Return an HttpClient using CHROMA_HOST / CHROMA_PORT from .env."""
    return chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", 8000)),
    )


def upsert(
    col: chromadb.Collection,
    ids: list[str],
    embeddings: np.ndarray,
    documents: list[str],
    meta: dict,
) -> None:
    """Upsert documents + embeddings into a collection."""
    col.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=[{**meta, "doc_id": d} for d in ids],
    )


def query(
    col: chromadb.Collection,
    embedding: np.ndarray,
    n: int = 3,
) -> chromadb.QueryResult:
    """Query collection by embedding vector, return top-n results."""
    return col.query(
        query_embeddings=[embedding.tolist()],
        n_results=n,
        include=["documents", "distances"],
    )
