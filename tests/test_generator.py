"""Tests for answer generation response normalization."""

from __future__ import annotations

from backend.generation import generator
from backend.generation.generator import generate_answer


def test_generate_answer_extracts_confidence_sources_and_counts_tokens(monkeypatch) -> None:
    model_answer = (
        "Signer verifies signed values.\n\n"
        "CONFIDENCE: HIGH\n"
        "SOURCES: [src/itsdangerous/signer.py:Signer, src/itsdangerous/signer.py:unsign]"
    )
    monkeypatch.setattr(generator, "llm_call", lambda *args, **kwargs: model_answer)
    monkeypatch.setattr(generator, "estimate_token_count", lambda text: 123)

    result = generate_answer(
        question="How does Signer work?",
        merged_context="context",
        query_type="semantic",
        vector_hits_used=2,
        graph_triples_used=1,
    )

    assert result["confidence"] == "HIGH"
    assert result["sources"] == ["src/itsdangerous/signer.py:Signer", "src/itsdangerous/signer.py:unsign"]
    assert result["query_type"] == "semantic"
    assert result["vector_chunks_used"] == 2
    assert result["graph_triples_used"] == 1
    assert result["estimated_tokens"] == 123


def test_generate_answer_defaults_to_low_confidence_when_missing_marker(monkeypatch) -> None:
    monkeypatch.setattr(generator, "llm_call", lambda *args, **kwargs: "No marker")
    monkeypatch.setattr(generator, "estimate_token_count", lambda text: 10)

    result = generate_answer("question", "context", "semantic", 0, 0)

    assert result["confidence"] == "LOW"
    assert result["sources"] == []
