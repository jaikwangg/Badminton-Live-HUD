"""TrackNetV3 worker with an 8-frame cache and asynchronous median background."""

from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Protocol

import numpy as np
from PIL import Image

from hud.bus.messages import Frame, HeartbeatMessage, ShuttleMessage
from hud.bus.shm_ring import SharedFrameRing
from hud.config import AppConfig, endpoint, load_config
from hud.models.shuttle import ResizedFrame, ShuttleTracker, create_shuttle_tracker

LOGGER = logging.getLogger(__name__)

# Defaults for MedianBackground; run() reads models.shuttle.background_sample_span_s and
# (once T01 adds it) background_warmup_span_s.
BACKGROUND_SAMPLE_SPAN_S = 10.0
BACKGROUND_WARMUP_SPAN_S = 2.0

Crop = tuple[float, float, float, float]


def parse_crop(value: object) -> Crop | None:
    """models.shuttle.crop: [x0, y0, x1, y1] as frame fractions, or null for the full frame."""
    if value is None:
        return None
    crop = tuple(float(v) for v in value)  # type: ignore[attr-defined]
    if len(crop) != 4 or not (0.0 <= crop[0] < crop[2] <= 1.0 and 0.0 <= crop[1] < crop[3] <= 1.0):
        raise ValueError(f"crop must be [x0, y0, x1, y1] fractions, x0 < x1 and y0 < y1: {value}")
    return crop  # type: ignore[return-value]


def crop_box(width: int, height: int, crop: Crop | None) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) of the crop in a width x height frame; never empty."""
    if crop is None:
        return 0, 0, width, height
    x0 = min(width - 1, round(crop[0] * width))
    y0 = min(height - 1, round(crop[1] * height))
    x1 = min(width, max(x0 + 1, round(crop[2] * width)))
    y1 = min(height, max(y0 + 1, round(crop[3] * height)))
    return x0, y0, x1, y1


def _bgr_crop(frame: Frame, crop: Crop | None) -> np.ndarray:
    bgr = np.frombuffer(frame.image, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    x0, y0, x1, y1 = crop_box(frame.width, frame.height, crop)
    return bgr[y0:y1, x0:x1]


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
    """Build an RGB median of time-spaced frames off the frame-processing thread."""

    def __init__(
        self,
        width: int,
        height: int,
        sample_count: int,
        update_interval_s: float,
        sample_span_s: float = BACKGROUND_SAMPLE_SPAN_S,
        *,
        warmup_span_s: float = BACKGROUND_WARMUP_SPAN_S,
        crop: Crop | None = None,
    ) -> None:
        if sample_count < 1 or update_interval_s <= 0 or sample_span_s < 0 or warmup_span_s < 0:
            raise ValueError("Background sample count and update interval must be positive")
        self.width = width
        self.height = height
        self.sample_count = sample_count
        self.update_interval_s = update_interval_s
        self.sample_span_s = sample_span_s
        self.warmup_span_s = warmup_span_s
        self.crop = crop
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
        images = [_bgr_crop(frame, self.crop) for frame in samples]
        median_bgr = np.median(np.stack(images, axis=0), axis=0).astype(np.uint8)
        median_rgb = median_bgr[:, :, ::-1]
        # Official TrackNetV3 uses PIL Image.resize with its default BICUBIC filter.
        resized = Image.fromarray(median_rgb).resize((self.width, self.height))
        return np.asarray(resized, dtype=np.uint8).copy()

    def _run(self) -> None:
        # Frames spaced sample_span_s / sample_count apart by capture ts, so players who
        # move within the span drop out of the median (consecutive frames kept them in).
        # Fast start: the first median spans only warmup_span_s; the samples then restart at
        # the full span and replace it as soon as they are complete ("widening").
        samples: deque[Frame] = deque(maxlen=self.sample_count)
        warming = self.warmup_span_s < self.sample_span_s
        widening = False
        last_update = 0.0
        source_shape: tuple[int, int] | None = None
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.1)
            except Empty:
                continue
            current_shape = (frame.height, frame.width)
            if source_shape is not None and current_shape != source_shape:
                samples.clear()
                self._background = None
                last_update = 0.0
                warming, widening = self.warmup_span_s < self.sample_span_s, False
            source_shape = current_shape

            span_s = self.warmup_span_s if warming else self.sample_span_s
            if samples and frame.ts - samples[-1].ts < int(span_s * 1e9 / self.sample_count):
                continue
            samples.append(frame)

            now = time.monotonic()
            if len(samples) < self.sample_count or (
                self._background is not None
                and not widening
                and now - last_update < self.update_interval_s
            ):
                continue
            try:
                self._background = self._make_background(list(samples))
                last_update = now
            except Exception:
                LOGGER.exception("Failed to update shuttle median background")
                continue
            if warming:
                samples.clear()
            widening, warming = warming, False


class ShuttleWorker:
    def __init__(
        self,
        tracker: ShuttleTracker,
        publisher: Publisher,
        *,
        max_gap: int = 1,
        background: MedianBackground | None = None,
        executor: Executor | None = None,
        crop: Crop | None = None,
    ) -> None:
        if max_gap < 0:
            raise ValueError("max_gap must be non-negative")
        self.tracker = tracker
        self.publisher = publisher
        self.max_gap = max_gap
        self.crop = crop
        self.background = background
        # None runs inference inline; run() passes the model-load thread so the frame
        # loop and its heartbeat never wait on a slow model.
        self.executor = executor
        self._pending: (
            tuple[ResizedFrame, Future[tuple[bool, list[float] | None, float | None, float]]] | None
        ) = None
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
        x0, y0, x1, y1 = crop_box(frame.width, frame.height, self.crop)
        rgb = _bgr_crop(frame, self.crop)[:, :, ::-1]
        # Match dataset.py:437-451, which loads RGB with PIL and calls Image.resize().
        resized = Image.fromarray(rgb).resize((self.tracker.width, self.tracker.height))
        return ResizedFrame(
            frame_id=frame.frame_id,
            ts=frame.ts,
            source_width=x1 - x0,
            source_height=y1 - y0,
            image_rgb=np.asarray(resized, dtype=np.uint8).copy(),
            origin_x=x0,
            origin_y=y0,
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

    def _publish(
        self,
        frame_id: int,
        ts: int,
        visible: bool = False,
        px: list[float] | None = None,
        confidence: float | None = None,
    ) -> None:
        self.publisher.send(
            ShuttleMessage(
                frame_id=frame_id, ts=ts, visible=visible, px=px, conf=confidence
            ).to_json()
        )
        self.processed += 1

    def _timed_infer(
        self, frames: tuple[ResizedFrame, ...], background: np.ndarray
    ) -> tuple[bool, list[float] | None, float | None, float]:
        started_ns = time.perf_counter_ns()
        visible, px, confidence = self.tracker.infer(frames, background)
        return visible, px, confidence, (time.perf_counter_ns() - started_ns) / 1_000_000

    def _start_inference(self, background: np.ndarray) -> None:
        frames = tuple(self.frames)
        if self.executor is None:
            future: Future[tuple[bool, list[float] | None, float | None, float]] = Future()
            try:
                future.set_result(self._timed_infer(frames, background))
            except Exception as exc:
                future.set_exception(exc)
        else:
            future = self.executor.submit(self._timed_infer, frames, background)
        self._pending = (frames[-1], future)

    def _collect(self) -> None:
        """Publish a finished inference; never waits."""
        if self._pending is not None and self._pending[1].done():
            frame, future = self._pending
            self._pending = None
            try:
                visible, px, confidence, self.last_proc_ms = future.result()
            except Exception:
                # A bad model output degrades shuttle tracking only.
                LOGGER.exception("Shuttle inference failed at frame %s", frame.frame_id)
                visible, px, confidence = False, None, None
            self._publish(frame.frame_id, frame.ts, visible, px, confidence)

    def poll(self) -> None:
        """Publish a finished inference and send a due heartbeat; never waits."""
        self._collect()
        if time.monotonic() - self.last_heartbeat >= 1.0:
            self._heartbeat()

    def process(self, frame: Frame) -> None:
        self.last_ts = frame.ts
        try:
            if self.background is not None:
                self.background.submit(frame)
            self._append_window(self._resize_frame(frame))
        except Exception:
            LOGGER.exception("Shuttle preprocessing failed at frame %s", frame.frame_id)
            self._publish(frame.frame_id, frame.ts)
            self.poll()
            return
        self._collect()
        if self._pending is None:
            background = self.background.get() if self.background is not None else None
            if len(self.frames) == self.tracker.seq_len and background is not None:
                # The window snapshot is always the newest cached frames.
                self._start_inference(background)
            else:
                # Not inferred: conf=null tells fusion this is not a "no shuttle" result.
                self._publish(frame.frame_id, frame.ts)
        # While inference is busy the frame is only cached, not published: a conf=null
        # message per skipped frame would count as a miss in fusion's Kalman filter.
        self.poll()


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
    publisher = ZmqPublisher(endpoint(config, "shuttle"))
    bus = SharedFrameRing(
        config.bus.name,
        config.source.width,
        config.source.height,
        config.bus.slots,
        create=False,
    )
    background: MedianBackground | None = None
    # One model thread loads, warms up and infers, so this thread only publishes; it also
    # keeps a TensorRT CUDA context on the thread that created it.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shuttle-model")
    try:
        future = pool.submit(_load_and_warmup, settings)
        _send_startup_heartbeat(publisher)
        while not future.done():
            _send_startup_heartbeat(publisher)
            future_done = Event()
            future_done.wait(1.0)
        tracker = future.result()
        crop = parse_crop(settings.get("crop"))

        background = MedianBackground(
            tracker.width,
            tracker.height,
            int(settings["background_sample_count"]),
            float(settings["background_update_interval_s"]),
            sample_span_s=float(settings["background_sample_span_s"]),
            warmup_span_s=float(settings.get("background_warmup_span_s", BACKGROUND_WARMUP_SPAN_S)),
            crop=crop,
        ).start()
        worker = ShuttleWorker(
            tracker,
            publisher,
            max_gap=int(settings["max_gap"]),
            background=background,
            executor=pool,
            crop=crop,
        )
        idle_wait = Event()
        while True:
            frame = bus.read_latest()
            if frame is None or (
                worker.last_frame_id is not None and frame.frame_id <= worker.last_frame_id
            ):
                worker.poll()
                idle_wait.wait(0.001)
                continue
            worker.process(frame)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        if background is not None:
            background.close()
        bus.close()
        publisher.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run(load_config())


if __name__ == "__main__":
    main()
