"""Read newest frames, track players, and publish contract v1 messages."""

from __future__ import annotations

import time
from threading import Event
from typing import Any, Protocol

import numpy as np

from hud.bus.messages import HeartbeatMessage, PlayersMessage
from hud.bus.shm_ring import SharedFrameRing
from hud.config import AppConfig, load_config
from hud.models.detector import Detector, create_detector


class Publisher(Protocol):
    def send(self, message: str) -> None: ...


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


class PlayersWorker:
    def __init__(
        self,
        detector: Detector,
        publisher: Publisher,
        *,
        roi: list[float] | None = None,
    ) -> None:
        self.detector = detector
        self.publisher = publisher
        self.roi = roi
        self._last_heartbeat = 0.0
        self._processed = 0
        self._started = time.monotonic()
        self._last_proc_ms: float | None = None
        self._last_ts = 0

    def _heartbeat(self) -> None:
        now = time.monotonic()
        message = HeartbeatMessage(
            worker="players",
            ts=self._last_ts,
            fps=self._processed / max(now - self._started, 1e-9),
            proc_ms=self._last_proc_ms,
        ).to_json()
        self.publisher.send(message)
        # The supervisor watches heartbeat lines on this worker's stdout.
        print(message, flush=True)
        self._last_heartbeat = now

    def _inside_roi(self, x: float, y: float, width: int, height: int) -> bool:
        if self.roi is None:
            return True
        left, top, right, bottom = self.roi
        return left * width <= x <= right * width and top * height <= y <= bottom * height

    def process(self, frame_id: int, ts: int, width: int, height: int, image: bytes) -> None:
        started_ns = time.perf_counter_ns()
        image_array = np.frombuffer(image, dtype=np.uint8).reshape(height, width, 3)
        detections = self.detector.infer(image_array)
        players: list[dict[str, Any]] = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            foot_x = (x1 + x2) / 2.0
            foot_y = y2
            if not self._inside_roi(foot_x, foot_y, width, height):
                continue
            if detection.track_id is None:
                continue
            players.append(
                {
                    "track_id": detection.track_id,
                    "bbox": detection.bbox,
                    "foot_px": [foot_x, foot_y],
                    "conf": detection.conf,
                }
            )
        self.publisher.send(PlayersMessage(frame_id=frame_id, ts=ts, players=players).to_json())
        self._processed += 1
        self._last_ts = ts
        self._last_proc_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        now = time.monotonic()
        if now - self._last_heartbeat >= 1.0:
            self._heartbeat()


def run(config: AppConfig) -> None:
    model_config = config.models["players"]
    detector = create_detector(model_config)
    publisher = ZmqPublisher("ipc:///tmp/hud_players")
    bus = SharedFrameRing(
        config.bus.name,
        config.source.width,
        config.source.height,
        config.bus.slots,
        create=False,
    )
    worker = PlayersWorker(detector, publisher, roi=model_config.get("roi"))
    try:
        idle_wait = Event()
        last_frame_id = 0
        while True:
            frame = bus.read_latest()
            if frame is None or frame.frame_id <= last_frame_id:
                if time.monotonic() - worker._last_heartbeat >= 1.0:
                    worker._heartbeat()
                idle_wait.wait(0.001)
                continue
            last_frame_id = frame.frame_id
            worker.process(frame.frame_id, frame.ts, frame.width, frame.height, frame.image)
    finally:
        bus.close()
        publisher.close()


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()
