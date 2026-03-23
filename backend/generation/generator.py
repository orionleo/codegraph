"""LLM answer generation from merged code retrieval context."""

from __future__ import annotations

import re
import os

from backend.config import PROVIDER, estimate_token_count, llm_call

SYSTEM_PROMPT = """You are an expert software engineer and code analyst.
Answer the question using ONLY the provided code context.
When referencing code, mention the exact file path and function name.
If asked about dependencies or impact, trace through the graph context carefully.
If the answer is not in the context, say "This information isn't in the indexed codebase."

Always end your answer with:
CONFIDENCE: HIGH | MEDIUM | LOW
SOURCES: [list of file_path:function_name references used]
"""
MAX_CONTEXT_CHARS = int(os.getenv("GEN_MAX_CONTEXT_CHARS", "18000"))


def generate_answer(
    question: str,
    merged_context: str,
    query_type: str,
    vector_hits_used: int,
    graph_triples_used: int,
    provider: str | None = None,
) -> dict[str, str | int]:
    """Generate an answer constrained to retrieved evidence."""

    bounded_context = merged_context[:MAX_CONTEXT_CHARS]
    prompt = f"Code Context:\n{bounded_context}\n\nQuestion: {question}"
    answer = llm_call(prompt, system_prompt=SYSTEM_PROMPT, provider=provider or PROVIDER).strip()
    return {
        "answer": answer,
        "confidence": _extract_confidence(answer),
        "sources": _extract_sources(answer),
        "query_type": query_type,
        "vector_chunks_used": vector_hits_used,
        "graph_triples_used": graph_triples_used,
        "estimated_tokens": estimate_token_count(prompt + answer),
    }


def _extract_confidence(answer_text: str) -> str:
    upper = answer_text.upper()
    for level in ("HIGH", "MEDIUM", "LOW"):
        if f"CONFIDENCE: {level}" in upper:
            return level
    return "LOW"


def _extract_sources(answer_text: str) -> list[str]:
    match = re.search(r"SOURCES:\s*\[(.*?)\]", answer_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    return [item.strip().strip("'\"") for item in match.group(1).split(",") if item.strip()]
