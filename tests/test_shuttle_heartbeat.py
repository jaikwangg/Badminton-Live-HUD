from __future__ import annotations

import json
from collections.abc import Sequence

from hud.bus.messages import Frame
from hud.workers.shuttle import ShuttleWorker


class NullTracker:
    def infer(self, frames: Sequence[Frame]) -> tuple[bool, list[float] | None, float | None]:
        return False, None, None


class NullPublisher:
    def send(self, message: str) -> None:
        pass

    def close(self) -> None:
        pass


def test_heartbeat_is_printed_for_supervisor(capsys) -> None:
    worker = ShuttleWorker(NullTracker(), NullPublisher())
    worker.process(Frame(1, 1000, 4, 2, bytes(4 * 2 * 3)))
    line = capsys.readouterr().out.strip()
    assert json.loads(line)["worker"] == "shuttle"
