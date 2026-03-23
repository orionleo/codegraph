"""Semantic vector retrieval over ChromaDB for code chunks."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from backend.config import DEFAULT_CHROMA_COLLECTION, PROVIDER, embed
from backend.pipeline.embedder import get_chroma_collection


def vector_search(
    query: str,
    top_k: int = 5,
    provider: str | None = None,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
) -> list[dict[str, Any]]:
    """Retrieve the nearest code chunk matches from ChromaDB."""

    if top_k <= 0:
        return []
    collection = get_chroma_collection(collection_name)
    query_embedding = embed(query, provider or PROVIDER)
    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    except Exception:
        return []

    matches: list[dict[str, Any]] = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for chunk_id, document, distance, metadata in zip(ids, documents, distances, metadatas):
        row = dict(metadata or {})
        row.update(
            {
                "chunk_id": str(chunk_id),
                "text": str(document),
                "distance": float(distance),
            }
        )
        matches.append(row)
    return sorted(matches, key=_rank_key)


def _rank_key(match: dict[str, Any]) -> tuple[float, float]:
    file_path = str(match.get("file_path", ""))
    penalty = _path_penalty(file_path)
    return (penalty, float(match.get("distance", 1.0)))


def _path_penalty(file_path: str) -> float:
    path = PurePosixPath(file_path.lower())
    parts = set(path.parts)
    if "src" in parts:
        return 0.0
    if "itsdangerous" in parts and "tests" not in parts:
        return 0.05
    if "tests" in parts or "test" in parts:
        return 0.8
    if "scripts" in parts:
        return 0.95
    if "docs" in parts or "docs_src" in parts or "examples" in parts:
        return 0.9
    return 0.2
