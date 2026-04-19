"""Tests for ingest progress tracking."""

from __future__ import annotations

from backend.progress import ProgressTracker


def test_progress_tracker_clamps_percent_and_preserves_details() -> None:
    tracker = ProgressTracker()

    tracker.reset("start")
    tracker.update("embedding", 150, "too high", chunks_processed=10)
    snapshot = tracker.snapshot()

    assert snapshot["phase"] == "embedding"
    assert snapshot["percent"] == 100
    assert snapshot["done"] is False
    assert snapshot["details"]["chunks_processed"] == 10

    tracker.fail("boom")
    failed = tracker.snapshot()
    assert failed["phase"] == "failed"
    assert failed["error"] == "boom"
    assert failed["done"] is True

    tracker.finish("ok", files_processed=3)
    finished = tracker.snapshot()
    assert finished["phase"] == "done"
    assert finished["percent"] == 100
    assert finished["details"]["files_processed"] == 3
