"""Capture loop that reconnects stalled sources and writes the latest frame to the bus."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from hud.bus.messages import HeartbeatMessage
from hud.bus.shm_ring import SharedFrameRing
from hud.config import AppConfig, load_config
from hud.sources.base import Source
from hud.sources.camera import CameraSource
from hud.sources.file_replay import FileReplaySource
from hud.sources.srt import SRTSource


def create_source(config: AppConfig) -> Source:
    """Create a configured source; switching ``source.type`` selects the backend."""
    settings = config.source
    if settings.type == "file" and settings.path:
        return FileReplaySource(settings.path, loop=settings.loop)
    if settings.type == "camera":
        device = int(settings.path) if settings.path is not None else 0
        return CameraSource(device, width=settings.width, height=settings.height)
    if settings.type == "srt" and settings.path:
        return SRTSource(settings.path, width=settings.width, height=settings.height)
    raise ValueError(
        f"Invalid source configuration: type={settings.type!r}, path={settings.path!r}"
    )


class CaptureProcess:
    def __init__(
        self,
        config: AppConfig,
        source_factory: Callable[[], Source] | None = None,
        *,
        reconnect_after_s: float = 1.0,
        max_backoff_s: float = 5.0,
    ) -> None:
        self.config = config
        self.source_factory = source_factory or (lambda: create_source(config))
        self.reconnect_after_s = reconnect_after_s
        self.max_backoff_s = max_backoff_s
        self._stop = False

    def run(self) -> None:
        bus = SharedFrameRing(
            self.config.bus.name,
            self.config.source.width,
            self.config.source.height,
            self.config.bus.slots,
        )
        source: Source | None = None
        frame_count = 0
        started_at = time.monotonic()
        last_frame_at = started_at
        last_heartbeat_at = started_at
        backoff = 0.25

        def heartbeat(ts: int | None = None) -> None:
            nonlocal last_heartbeat_at
            now = time.monotonic()
            if now - last_heartbeat_at < 1.0:
                return
            print(
                HeartbeatMessage(
                    worker="capture",
                    ts=ts if ts is not None else time.monotonic_ns(),
                    fps=frame_count / max(now - started_at, 1e-9),
                    proc_ms=None,
                ).to_json(),
                flush=True,
            )
            last_heartbeat_at = now

        def wait_backoff() -> None:
            # Backoff can exceed the supervisor's 3 s heartbeat timeout, so keep beating.
            nonlocal backoff
            deadline = time.monotonic() + backoff
            while not self._stop and (remaining := deadline - time.monotonic()) > 0:
                time.sleep(min(remaining, 0.5))
                heartbeat()
            backoff = min(backoff * 2, self.max_backoff_s)

        try:
            while not self._stop:
                if source is None:
                    try:
                        source = self.source_factory()
                        source.open()
                        last_frame_at = time.monotonic()
                    except Exception:
                        source = None
                        wait_backoff()
                        continue
                try:
                    frame = source.read()
                except Exception:
                    frame = None
                now = time.monotonic()
                if frame is not None:
                    image = np.frombuffer(frame.image, dtype=np.uint8).reshape(
                        frame.height, frame.width, 3
                    )
                    if image.shape[:2] != (bus.height, bus.width):
                        # Sources may change resolution (camera reconnect, other file).
                        import cv2

                        image = cv2.resize(image, (bus.width, bus.height))
                    bus.write(image, frame.ts)
                    frame_count += 1
                    last_frame_at = now
                    backoff = 0.25
                elif now - last_frame_at > self.reconnect_after_s:
                    source.close()
                    source = None
                    wait_backoff()
                heartbeat(frame.ts if frame is not None else None)
        finally:
            if source is not None:
                source.close()
            bus.close()

    def stop(self) -> None:
        self._stop = True


def main() -> None:
    CaptureProcess(load_config()).run()


if __name__ == "__main__":
    main()
