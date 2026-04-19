"""Tests for query classification fallback behavior."""

from __future__ import annotations

from backend.retrieval import query_classifier
from backend.retrieval.query_classifier import classify_query


def test_classify_query_falls_back_to_definition_when_llm_fails(monkeypatch) -> None:
    monkeypatch.setattr(query_classifier, "llm_call", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    result = classify_query("Where is Signer defined?")

    assert result.type == "definition"
    assert result.entity == "Signer"
    assert result.graph_limit == 10


def test_classify_query_falls_back_to_impact_and_ignores_stopwords(monkeypatch) -> None:
    monkeypatch.setattr(query_classifier, "llm_call", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    result = classify_query("What breaks if I change Serializer?")

    assert result.type == "impact"
    assert result.entity == "Serializer"


def test_classify_query_uses_valid_llm_json(monkeypatch) -> None:
    monkeypatch.setattr(
        query_classifier,
        "llm_call",
        lambda *args, **kwargs: '{"type": "dependency", "entity": "unsign"}',
    )

    result = classify_query("What calls unsign?")

    assert result.type == "dependency"
    assert result.entity == "unsign"
