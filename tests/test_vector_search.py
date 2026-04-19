"""Tests for vector result ranking and path penalties."""

from __future__ import annotations

from backend.retrieval.vector_search import _path_penalty, _rank_key


def test_path_penalty_prefers_source_over_tests_docs_and_scripts() -> None:
    assert _path_penalty("src/itsdangerous/signer.py") < _path_penalty("tests/test_signer.py")
    assert _path_penalty("src/itsdangerous/signer.py") < _path_penalty("docs/conf.py")
    assert _path_penalty("scripts/release.py") > _path_penalty("tests/test_signer.py")


def test_rank_key_uses_path_penalty_before_distance() -> None:
    source_hit = {"file_path": "src/pkg/core.py", "distance": 0.9}
    test_hit = {"file_path": "tests/test_core.py", "distance": 0.1}

    assert _rank_key(source_hit) < _rank_key(test_hit)
