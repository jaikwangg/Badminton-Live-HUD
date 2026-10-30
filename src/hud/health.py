"""Lightweight health status primitives for worker supervision."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic


@dataclass(slots=True)
class WorkerHealth:
    name: str
    last_heartbeat: float = field(default_factory=monotonic)
    status: str = "ok"

    def beat(self) -> None:
        self.last_heartbeat = monotonic()
        self.status = "ok"

    def check(self, timeout_s: float = 3.0) -> str:
        if monotonic() - self.last_heartbeat > timeout_s:
            self.status = "down"
        return self.status
