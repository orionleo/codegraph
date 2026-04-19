"""FastAPI route tests with external services mocked."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from backend import main


@dataclass(slots=True)
class FakeClassification:
    type: str = "definition"
    entity: str = "Signer"


async def fake_retrieval(question: str) -> dict[str, object]:
    return {
        "classification": FakeClassification(),
        "context": "context",
        "vector_hits": [{"chunk_id": "src/signer.py::Signer"}],
        "graph_triples": ["Class Signer defined in src/signer.py:76"],
        "graph_records": [{"name": "Signer"}],
        "metadata": {"vector_hits_count": 1, "graph_triples_count": 1},
    }


def fake_answer(**kwargs) -> dict[str, object]:
    return {
        "answer": "Signer is defined in source.\n\nCONFIDENCE: HIGH\nSOURCES: [src/signer.py:Signer]",
        "confidence": "HIGH",
        "sources": ["src/signer.py:Signer"],
        "query_type": kwargs["query_type"],
        "vector_chunks_used": kwargs["vector_hits_used"],
        "graph_triples_used": kwargs["graph_triples_used"],
        "estimated_tokens": 42,
    }


def test_query_endpoint_returns_grounded_answer_payload(monkeypatch) -> None:
    monkeypatch.setattr(main, "merge_hybrid_context", fake_retrieval)
    monkeypatch.setattr(main, "generate_answer", fake_answer)
    client = TestClient(main.app)

    response = client.post("/query", json={"question": "Where is Signer defined?", "mode": "graphrag"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["confidence"] == "HIGH"
    assert payload["entity"] == "Signer"
    assert payload["mode"] == "graphrag"
    assert payload["vector_chunks_used"] == 1
    assert payload["graph_triples_used"] == 1


def test_ingest_progress_endpoint_returns_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(
        main.INGEST_PROGRESS,
        "snapshot",
        lambda: {"phase": "done", "message": "ok", "percent": 100, "done": True, "error": None, "details": {}},
    )
    client = TestClient(main.app)

    response = client.get("/ingest/progress")

    assert response.status_code == 200
    assert response.json()["percent"] == 100
