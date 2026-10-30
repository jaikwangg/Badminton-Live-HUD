from __future__ import annotations

import json
import sys
import types
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from hud.bus.messages import Frame, ShuttleMessage
from hud.config import load_config
from hud.models.shuttle import ResizedFrame, TensorInfo, TrackNetV3Tracker
from hud.workers import shuttle as shuttle_worker
from hud.workers.shuttle import ShuttleWorker


class FakeSession:
    def __init__(self, output: np.ndarray | None = None) -> None:
        self.output = output if output is not None else np.zeros((1, 8, 288, 512), dtype=np.float32)
        self.last_values: dict[str, np.ndarray] = {}
        self.run_count = 0

    def get_inputs(self) -> list[TensorInfo]:
        return [TensorInfo("background_and_frames", (1, 27, 288, 512))]

    def get_outputs(self) -> list[TensorInfo]:
        return [TensorInfo("heatmaps", (1, 8, 288, 512))]

    def run(
        self, output_names: list[str] | None, values: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        self.last_values = values
        self.run_count += 1
        return [self.output]


class FakeTracker:
    seq_len = 8
    width = 32
    height = 18

    def __init__(self, *, raise_error: bool = False) -> None:
        self.calls: list[tuple[Sequence[ResizedFrame], np.ndarray]] = []
        self.raise_error = raise_error

    def warmup(self) -> None:
        pass

    def infer(
        self, frames: Sequence[ResizedFrame], background: np.ndarray
    ) -> tuple[bool, list[float] | None, float | None]:
        if self.raise_error:
            raise RuntimeError("synthetic inference error")
        self.calls.append((frames, background))
        return True, [12.0, 9.0], 0.8


class FixedBackground:
    def __init__(self, image: np.ndarray | None) -> None:
        self.image = image

    def submit(self, frame: Frame) -> None:
        pass

    def get(self) -> np.ndarray | None:
        return self.image


class FakePublisher:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, message: str) -> None:
        self.messages.append(message)

    def close(self) -> None:
        pass


def _resized_frame(
    frame_id: int,
    *,
    value: int = 0,
    source_width: int = 1920,
    source_height: int = 1080,
) -> ResizedFrame:
    return ResizedFrame(
        frame_id=frame_id,
        ts=frame_id * 1000,
        source_width=source_width,
        source_height=source_height,
        image_rgb=np.full((288, 512, 3), value, dtype=np.uint8),
    )


def _raw_frame(frame_id: int, width: int = 64, height: int = 48) -> Frame:
    return Frame(
        frame_id=frame_id,
        ts=frame_id * 1000,
        width=width,
        height=height,
        image=bytes(width * height * 3),
    )


def test_model_input_is_background_then_eight_rgb_frames_in_27_channels() -> None:
    session = FakeSession()
    tracker = TrackNetV3Tracker("unused.onnx", "onnxruntime", session=session, morph_kernel_size=1)
    background = np.zeros((288, 512, 3), dtype=np.uint8)
    background[0, 0] = [255, 128, 0]
    images = [np.full((288, 512, 3), 10 * (i + 1), dtype=np.uint8) for i in range(8)]
    images[0][0, 0] = [0, 64, 128]
    images[-1][0, 0] = [24, 48, 96]
    frames = [
        ResizedFrame(i + 1, (i + 1) * 1000, 1920, 1080, image) for i, image in enumerate(images)
    ]

    tensor = tracker._input_tensor(frames, background)

    assert tensor.shape == (1, 27, 288, 512)
    np.testing.assert_allclose(tensor[0, :3, 0, 0], [1.0, 128 / 255, 0.0])
    np.testing.assert_allclose(tensor[0, 3:6, 0, 0], [0.0, 64 / 255, 128 / 255])
    np.testing.assert_allclose(tensor[0, 24:27, 0, 0], [24 / 255, 48 / 255, 96 / 255])


def _reference_input_tensor(frames: Sequence[ResizedFrame], background: np.ndarray) -> np.ndarray:
    """The pre-cache implementation: transpose and concatenate all 9 images per call."""
    tensors = [background.transpose(2, 0, 1)]
    tensors += [frame.image_rgb.transpose(2, 0, 1) for frame in frames]
    return np.concatenate(tensors, axis=0)[None, ...].astype(np.float32) / 255.0


def test_cached_input_tensor_matches_reference_on_random_frames() -> None:
    rng = np.random.default_rng(6)
    tracker = TrackNetV3Tracker("unused.onnx", "onnxruntime", session=FakeSession())
    background = rng.integers(0, 256, (288, 512, 3), dtype=np.uint8)
    for step in range(3):
        if step == 2:
            # A new background array must replace the cached background channels.
            background = rng.integers(0, 256, (288, 512, 3), dtype=np.uint8)
        frames = [
            ResizedFrame(i, i, 1920, 1080, rng.integers(0, 256, (288, 512, 3), dtype=np.uint8))
            for i in range(8)
        ]
        np.testing.assert_allclose(
            tracker._input_tensor(frames, background),
            _reference_input_tensor(frames, background),
            rtol=0,
            atol=1e-7,
        )


def test_background_median_uses_frames_spaced_across_the_span() -> None:
    background = shuttle_worker.MedianBackground(8, 4, 3, 60.0, sample_span_s=3.0).start()
    try:
        # One frame every 0.5 s; the 1 s stride keeps frames 0, 2 and 4 (values 0, 20, 40).
        for index in range(5):
            frame = Frame(
                frame_id=index,
                ts=index * 500_000_000,
                width=8,
                height=4,
                image=bytes([index * 10]) * (8 * 4 * 3),
            )
            background.submit(frame)
            while not background._frames.empty():
                Event().wait(0.001)
        for _ in range(2000):
            if background.get() is not None:
                break
            Event().wait(0.001)
        image = background.get()
    finally:
        background.close()

    assert image is not None
    assert int(image[0, 0, 0]) == 20  # Consecutive frames 0, 1, 2 would give 10.


def test_infer_uses_last_of_eight_heatmaps_and_maps_pixel_centers() -> None:
    heatmaps = np.zeros((1, 8, 288, 512), dtype=np.float32)
    heatmaps[0, 0, 10:16, 10:16] = 0.99  # An earlier heatmap must be ignored.
    heatmaps[0, 7, 50:55, 100:105] = 0.9
    session = FakeSession(heatmaps)
    tracker = TrackNetV3Tracker(
        "unused.onnx",
        "onnxruntime",
        confidence_threshold=0.45,
        morph_kernel_size=1,
        min_contour_area=5,
        max_contour_area=120,
        session=session,
    )

    visible, px, confidence = tracker.infer(
        [_resized_frame(i + 1) for i in range(8)],
        np.zeros((288, 512, 3), dtype=np.uint8),
    )

    assert visible is True
    assert confidence == pytest.approx(0.9)
    assert px == pytest.approx([(102 + 0.5) * 1920 / 512, (52 + 0.5) * 1080 / 288])


def test_missing_background_emits_null_confidence_without_inference() -> None:
    tracker, publisher = FakeTracker(), FakePublisher()
    worker = ShuttleWorker(
        tracker,
        publisher,
        max_gap=1,
        background=FixedBackground(None),
    )

    for frame_id in range(1, 9):
        worker.process(_raw_frame(frame_id))

    message = ShuttleMessage.from_json(publisher.messages[-1])
    assert message.visible is False
    assert message.conf is None
    assert tracker.calls == []


def test_worker_fills_small_gap_and_resets_after_max_gap() -> None:
    tracker = FakeTracker()
    publisher = FakePublisher()
    worker = ShuttleWorker(
        tracker,
        publisher,
        max_gap=1,
        background=FixedBackground(np.zeros((18, 32, 3), dtype=np.uint8)),
    )
    for frame_id in range(1, 9):
        worker.process(_raw_frame(frame_id))
    assert len(tracker.calls) == 1

    worker.process(_raw_frame(10))  # Frame 9 is filled with the last cached image.
    assert len(tracker.calls) == 2
    assert [frame.frame_id for frame in tracker.calls[-1][0]][-3:] == [8, 8, 10]

    worker.process(_raw_frame(20))  # Missing frames exceed max_gap; window starts over.
    assert len(tracker.calls) == 2
    assert [frame.frame_id for frame in worker.frames] == [20]


def test_infer_error_emits_lost_message_and_worker_keeps_running() -> None:
    tracker = FakeTracker(raise_error=True)
    publisher = FakePublisher()
    worker = ShuttleWorker(
        tracker,
        publisher,
        background=FixedBackground(np.zeros((18, 32, 3), dtype=np.uint8)),
    )
    for frame_id in range(1, 8):
        worker.process(_raw_frame(frame_id))
    worker.last_heartbeat = 0.0
    worker.process(_raw_frame(8))

    message = ShuttleMessage.from_json(publisher.messages[-2])
    assert message.visible is False
    assert message.conf is None
    assert worker.processed == 8
    assert '"type":"heartbeat"' in publisher.messages[-1]


class SlowTracker(FakeTracker):
    """Blocks inference until released, like CPU fallback running slower than the camera."""

    def __init__(self) -> None:
        super().__init__()
        self.release = Event()

    def infer(
        self, frames: Sequence[ResizedFrame], background: np.ndarray
    ) -> tuple[bool, list[float] | None, float | None]:
        self.release.wait(5.0)
        return super().infer(frames, background)


def test_slow_inference_keeps_heartbeat_and_uses_newest_window() -> None:
    tracker, publisher = SlowTracker(), FakePublisher()
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = ShuttleWorker(
            tracker,
            publisher,
            max_gap=1,
            background=FixedBackground(np.zeros((18, 32, 3), dtype=np.uint8)),
            executor=pool,
        )
        for frame_id in range(1, 13):
            worker.process(_raw_frame(frame_id))
        worker.last_heartbeat = 0.0
        worker.process(_raw_frame(13))

        messages = [json.loads(message) for message in publisher.messages]
        shuttle = [message for message in messages if message["type"] == "shuttle"]
        # Frames 1-7 were not inferred; 9-13 were cached while frame 8 was still running.
        assert [message["frame_id"] for message in shuttle] == list(range(1, 8))
        assert all(message["conf"] is None for message in shuttle)
        assert messages[-1]["type"] == "heartbeat"

        tracker.release.set()
        assert worker._pending is not None
        worker._pending[1].result(timeout=5.0)
        worker.process(_raw_frame(14))
        worker._pending[1].result(timeout=5.0)

    result = ShuttleMessage.from_json(publisher.messages[-1])
    assert (result.frame_id, result.visible, result.conf) == (8, True, 0.8)
    assert [frame.frame_id for frame in tracker.calls[1][0]] == list(range(7, 15))


class ProviderSession(FakeSession):
    def __init__(self, path: str, providers: list[str]) -> None:
        super().__init__()
        self.requested = providers

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]


def test_cuda_fallback_to_cpu_marks_tracker_degraded(monkeypatch, caplog) -> None:
    fake_ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=ProviderSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    tracker = TrackNetV3Tracker("model.onnx", "onnxruntime")

    assert tracker.degraded is True
    assert "CUDAExecutionProvider" in tracker.session.requested
    assert "runs on ['CPUExecutionProvider']" in caplog.text


def test_run_binds_the_configured_bus_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config()
    config.bus.endpoints["shuttle"] = "tcp://127.0.0.1:59998"
    bound: list[str] = []

    def fake_publisher(address: str) -> None:
        bound.append(address)
        raise RuntimeError("stop")

    monkeypatch.setattr(shuttle_worker, "ZmqPublisher", fake_publisher)
    with pytest.raises(RuntimeError, match="stop"):
        shuttle_worker.run(config)
    assert bound == ["tcp://127.0.0.1:59998"]


def test_run_passes_the_configured_background_sample_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config()
    config.models["shuttle"]["background_sample_span_s"] = 4.5
    spans: list[float] = []

    class Closable:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        def close(self) -> None: ...

    tracker = types.SimpleNamespace(width=512, height=288)

    def fake_background(*args: object, sample_span_s: float) -> None:
        spans.append(sample_span_s)
        raise RuntimeError("stop")

    monkeypatch.setattr(shuttle_worker, "ZmqPublisher", Closable)
    monkeypatch.setattr(shuttle_worker, "SharedFrameRing", Closable)
    monkeypatch.setattr(shuttle_worker, "_send_startup_heartbeat", lambda publisher: None)
    monkeypatch.setattr(shuttle_worker, "_load_and_warmup", lambda settings: tracker)
    monkeypatch.setattr(shuttle_worker, "MedianBackground", fake_background)
    with pytest.raises(RuntimeError, match="stop"):
        shuttle_worker.run(config)
    assert spans == [4.5]
