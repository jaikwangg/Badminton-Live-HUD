"""TrackNet shuttle worker with a resettable three-frame sliding window."""

from __future__ import annotations

import time
from collections import deque
from threading import Event
from typing import Protocol

from hud.bus.messages import Frame, HeartbeatMessage, ShuttleMessage
from hud.bus.shm_ring import SharedFrameRing
from hud.config import AppConfig, load_config
from hud.models.shuttle import ShuttleTracker, create_shuttle_tracker


class Publisher(Protocol):
    def send(self, message: str) -> None: ...

    def close(self) -> None: ...


class ZmqPublisher:
    def __init__(self, endpoint: str) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("Install pyzmq to publish worker messages") from exc
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 4)
        self._socket.bind(endpoint)

    def send(self, message: str) -> None:
        self._socket.send_string(message)

    def close(self) -> None:
        self._socket.close(linger=0)


class ShuttleWorker:
    def __init__(self, tracker: ShuttleTracker, publisher: Publisher) -> None:
        self.tracker = tracker
        self.publisher = publisher
        self.frames: deque[Frame] = deque(maxlen=3)
        self.last_frame_id: int | None = None
        self.last_heartbeat = 0.0
        self.started = time.monotonic()
        self.processed = 0
        self.last_proc_ms: float | None = None
        self.last_ts = 0

    def _heartbeat(self) -> None:
        now = time.monotonic()
        self.publisher.send(
            HeartbeatMessage(
                worker="shuttle",
                ts=self.last_ts,
                fps=self.processed / max(now - self.started, 1e-9),
                proc_ms=self.last_proc_ms,
            ).to_json()
        )
        self.last_heartbeat = now

    def process(self, frame: Frame) -> None:
        if self.last_frame_id is not None and frame.frame_id != self.last_frame_id + 1:
            self.frames.clear()
        self.last_frame_id = frame.frame_id
        self.last_ts = frame.ts
        self.frames.append(frame)
        started_ns = time.perf_counter_ns()
        if len(self.frames) == 3:
            visible, px, confidence = self.tracker.infer(tuple(self.frames))
        else:
            visible, px, confidence = False, None, None
        self.last_proc_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        self.publisher.send(
            ShuttleMessage(
                frame_id=frame.frame_id,
                ts=frame.ts,
                visible=visible,
                px=px,
                conf=confidence,
            ).to_json()
        )
        self.processed += 1
        if time.monotonic() - self.last_heartbeat >= 1.0:
            self._heartbeat()


def run(config: AppConfig) -> None:
    settings = config.models["shuttle"]
    tracker = create_shuttle_tracker(settings)
    publisher = ZmqPublisher("ipc:///tmp/hud_shuttle")
    bus = SharedFrameRing(
        config.bus.name,
        config.source.width,
        config.source.height,
        config.bus.slots,
        create=False,
    )
    worker = ShuttleWorker(tracker, publisher)
    idle_wait = Event()
    try:
        while True:
            frame = bus.read_latest()
            if frame is None or (
                worker.last_frame_id is not None and frame.frame_id <= worker.last_frame_id
            ):
                if time.monotonic() - worker.last_heartbeat >= 1.0:
                    worker._heartbeat()
                idle_wait.wait(0.001)
                continue
            worker.process(frame)
    finally:
        bus.close()
        publisher.close()


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()
