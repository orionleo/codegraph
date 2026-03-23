"""In-memory progress tracking for long-running ingest operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass(slots=True)
class ProgressState:
    """Mutable ingest progress state."""

    phase: str = "idle"
    message: str = "Idle."
    percent: int = 0
    done: bool = True
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """Thread-safe progress tracker for the current ingest job."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._state = ProgressState()

    def reset(self, message: str = "Starting ingest.") -> None:
        with self._lock:
            self._state = ProgressState(
                phase="starting",
                message=message,
                percent=0,
                done=False,
                error=None,
                details={},
            )

    def update(self, phase: str, percent: int, message: str, **details: Any) -> None:
        with self._lock:
            self._state.phase = phase
            self._state.percent = max(0, min(int(percent), 100))
            self._state.message = message
            self._state.done = False
            self._state.error = None
            if details:
                self._state.details.update(details)

    def fail(self, message: str) -> None:
        with self._lock:
            self._state.phase = "failed"
            self._state.message = message
            self._state.error = message
            self._state.done = True

    def finish(self, message: str, **details: Any) -> None:
        with self._lock:
            self._state.phase = "done"
            self._state.percent = 100
            self._state.message = message
            self._state.done = True
            self._state.error = None
            if details:
                self._state.details.update(details)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "phase": self._state.phase,
                "message": self._state.message,
                "percent": self._state.percent,
                "done": self._state.done,
                "error": self._state.error,
                "details": dict(self._state.details),
            }


INGEST_PROGRESS = ProgressTracker()
