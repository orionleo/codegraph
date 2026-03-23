"""Parallel hybrid retrieval and merged context formatting."""

from __future__ import annotations

import asyncio
from typing import Any

from backend.retrieval.graph_search import graph_search
from backend.retrieval.query_classifier import QueryClassification, classify_query
from backend.retrieval.vector_search import vector_search


async def _run_vector_search(question: str, top_k: int) -> list[dict[str, Any]]:
    return await asyncio.to_thread(vector_search, question, top_k)


async def _run_graph_search(classification: QueryClassification) -> dict[str, Any]:
    return await asyncio.to_thread(graph_search, classification)


async def merge_hybrid_context(question: str) -> dict[str, Any]:
    """Retrieve vector and graph context concurrently using query-aware weighting."""

    classification = classify_query(question)
    vector_hits, graph_payload = await asyncio.gather(
        _run_vector_search(question, classification.vector_limit),
        _run_graph_search(classification),
    )

    graph_lines = graph_payload["formatted"][: classification.graph_limit]
    vector_hits = vector_hits[: classification.vector_limit]
    merged_context = _format_context(vector_hits, graph_lines)
    return {
        "classification": classification,
        "context": merged_context,
        "vector_hits": vector_hits,
        "graph_triples": graph_lines,
        "graph_records": graph_payload["records"],
        "metadata": {
            "vector_hits_count": len(vector_hits),
            "graph_triples_count": len(graph_lines),
        },
    }


async def merge_vector_only_context(question: str) -> dict[str, Any]:
    """Build the vanilla RAG baseline context using vectors only."""

    classification = classify_query(question)
    vector_hits = await _run_vector_search(question, max(8, classification.vector_limit or 8))
    context = _format_context(vector_hits, [])
    return {
        "classification": classification,
        "context": context,
        "vector_hits": vector_hits,
        "graph_triples": [],
        "graph_records": [],
        "metadata": {
            "vector_hits_count": len(vector_hits),
            "graph_triples_count": 0,
        },
    }


def _format_context(vector_hits: list[dict[str, Any]], graph_lines: list[str]) -> str:
    semantic_lines = []
    file_lines = []
    for hit in vector_hits:
        chunk_type = hit.get("type", "function")
        if chunk_type == "file_summary":
            file_lines.append(f"{hit.get('file_path')}: {hit.get('text')}")
            continue
        signature = hit.get("signature") or ""
        semantic_lines.append(
            f"[{chunk_type.title()}: {hit.get('name')} in {hit.get('file_path')}:{hit.get('line_start')}]\n"
            f"{signature}\n{hit.get('text')}"
        )

    return (
        "=== Semantic Code Context ===\n"
        f"{chr(10).join(semantic_lines) or 'No semantic matches.'}\n\n"
        "=== Graph Context ===\n"
        f"{chr(10).join(graph_lines) or 'No graph matches.'}\n\n"
        "=== File Context ===\n"
        f"{chr(10).join(file_lines) or 'No file summaries.'}"
    )
