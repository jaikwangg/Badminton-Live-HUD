"""TrackNetV3 worker with an 8-frame cache and asynchronous median background."""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Protocol

import numpy as np
from PIL import Image

from hud.bus.messages import Frame, HeartbeatMessage, ShuttleMessage
from hud.bus.shm_ring import SharedFrameRing
from hud.config import AppConfig, load_config
from hud.models.shuttle import ResizedFrame, ShuttleTracker, create_shuttle_tracker

LOGGER = logging.getLogger(__name__)


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


class MedianBackground:
    """Build a reservoir-sampled RGB median off the frame-processing thread."""

    def __init__(
        self,
        width: int,
        height: int,
        sample_count: int,
        update_interval_s: float,
    ) -> None:
        if sample_count < 1 or update_interval_s <= 0:
            raise ValueError("Background sample count and update interval must be positive")
        self.width = width
        self.height = height
        self.sample_count = sample_count
        self.update_interval_s = update_interval_s
        self._frames: Queue[Frame] = Queue(maxsize=1)
        self._stop = Event()
        self._thread: Thread | None = None
        self._background: np.ndarray | None = None

    def start(self) -> MedianBackground:
        self._thread = Thread(target=self._run, name="shuttle-background", daemon=True)
        self._thread.start()
        return self

    def submit(self, frame: Frame) -> None:
        """Queue only the newest frame; never wait for the background thread."""
        try:
            self._frames.put_nowait(frame)
            return
        except Full:
            pass
        try:
            self._frames.get_nowait()
        except Empty:
            pass
        try:
            self._frames.put_nowait(frame)
        except Full:
            # The consumer won the race and the incoming frame is already obsolete.
            return

    def get(self) -> np.ndarray | None:
        return self._background

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _make_background(self, samples: list[Frame]) -> np.ndarray:
        images = [
            np.frombuffer(frame.image, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            for frame in samples
        ]
        median_bgr = np.median(np.stack(images, axis=0), axis=0).astype(np.uint8)
        median_rgb = median_bgr[:, :, ::-1]
        # Official TrackNetV3 uses PIL Image.resize with its default BICUBIC filter.
        resized = Image.fromarray(median_rgb).resize((self.width, self.height))
        return np.asarray(resized, dtype=np.uint8).copy()

    def _run(self) -> None:
        rng = random.Random()
        reservoir: list[Frame] = []
        seen = 0
        last_update = 0.0
        source_shape: tuple[int, int] | None = None
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.1)
            except Empty:
                continue
            current_shape = (frame.height, frame.width)
            if source_shape is not None and current_shape != source_shape:
                reservoir.clear()
                seen = 0
                self._background = None
                last_update = 0.0
            source_shape = current_shape

            seen += 1
            if len(reservoir) < self.sample_count:
                reservoir.append(frame)
            else:
                replacement = rng.randrange(seen)
                if replacement < self.sample_count:
                    reservoir[replacement] = frame

            now = time.monotonic()
            if len(reservoir) < self.sample_count or (
                self._background is not None and now - last_update < self.update_interval_s
            ):
                continue
            try:
                self._background = self._make_background(reservoir.copy())
                last_update = now
            except Exception:
                LOGGER.exception("Failed to update shuttle median background")


class ShuttleWorker:
    def __init__(
        self,
        tracker: ShuttleTracker,
        publisher: Publisher,
        *,
        max_gap: int = 1,
        background: MedianBackground | None = None,
    ) -> None:
        if max_gap < 0:
            raise ValueError("max_gap must be non-negative")
        self.tracker = tracker
        self.publisher = publisher
        self.max_gap = max_gap
        self.background = background
        self.frames: deque[ResizedFrame] = deque(maxlen=tracker.seq_len)
        self.last_frame_id: int | None = None
        self.last_heartbeat = 0.0
        self.started = time.monotonic()
        self.processed = 0
        self.last_proc_ms: float | None = None
        self.last_ts = 0

    def _heartbeat(self) -> None:
        now = time.monotonic()
        message = HeartbeatMessage(
            worker="shuttle",
            ts=self.last_ts,
            fps=self.processed / max(now - self.started, 1e-9),
            proc_ms=self.last_proc_ms,
        ).to_json()
        self.publisher.send(message)
        print(message, flush=True)
        self.last_heartbeat = now

    def _resize_frame(self, frame: Frame) -> ResizedFrame:
        bgr = np.frombuffer(frame.image, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        rgb = bgr[:, :, ::-1]
        # Match dataset.py:437-451, which loads RGB with PIL and calls Image.resize().
        resized = Image.fromarray(rgb).resize((self.tracker.width, self.tracker.height))
        return ResizedFrame(
            frame_id=frame.frame_id,
            ts=frame.ts,
            source_width=frame.width,
            source_height=frame.height,
            image_rgb=np.asarray(resized, dtype=np.uint8).copy(),
        )

    def _append_window(self, current: ResizedFrame) -> None:
        if self.last_frame_id is not None:
            delta = current.frame_id - self.last_frame_id
            missing = delta - 1
            if delta <= 0 or missing > self.max_gap:
                self.frames.clear()
            elif missing > 0 and self.frames:
                previous = self.frames[-1]
                for _ in range(missing):
                    self.frames.append(previous)
        self.frames.append(current)
        self.last_frame_id = current.frame_id

    def process(self, frame: Frame) -> None:
        self.last_ts = frame.ts
        started_ns = time.perf_counter_ns()
        visible, px, confidence = False, None, None
        try:
            if self.background is not None:
                self.background.submit(frame)
            resized = self._resize_frame(frame)
            self._append_window(resized)
            background = self.background.get() if self.background is not None else None
            if len(self.frames) == self.tracker.seq_len and background is not None:
                visible, px, confidence = self.tracker.infer(tuple(self.frames), background)
        except Exception:
            # A bad model output or preprocessing frame degrades shuttle tracking only.
            LOGGER.exception("Shuttle inference failed at frame %s", frame.frame_id)
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


def _load_and_warmup(settings: dict[str, object]) -> ShuttleTracker:
    tracker = create_shuttle_tracker(settings)
    tracker.warmup()
    return tracker


def _send_startup_heartbeat(publisher: Publisher) -> None:
    message = HeartbeatMessage(worker="shuttle", ts=0, fps=None, proc_ms=None).to_json()
    publisher.send(message)
    print(message, flush=True)


def run(config: AppConfig) -> None:
    settings = config.models["shuttle"]
    publisher = ZmqPublisher("ipc:///tmp/hud_shuttle")
    bus = SharedFrameRing(
        config.bus.name,
        config.source.width,
        config.source.height,
        config.bus.slots,
        create=False,
    )
    background: MedianBackground | None = None
    try:
        # Keep publishing from this thread while model loading and warmup run elsewhere.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="shuttle-model-load") as pool:
            future = pool.submit(_load_and_warmup, settings)
            _send_startup_heartbeat(publisher)
            while not future.done():
                _send_startup_heartbeat(publisher)
                future_done = Event()
                future_done.wait(1.0)
            tracker = future.result()

        background = MedianBackground(
            tracker.width,
            tracker.height,
            int(settings["background_sample_count"]),
            float(settings["background_update_interval_s"]),
        ).start()
        worker = ShuttleWorker(
            tracker,
            publisher,
            max_gap=int(settings["max_gap"]),
            background=background,
        )
        idle_wait = Event()
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
        if background is not None:
            background.close()
        bus.close()
        publisher.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run(load_config())


if __name__ == "__main__":
    main()
