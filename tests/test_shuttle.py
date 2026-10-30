from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from hud.bus.messages import Frame, ShuttleMessage
from hud.models.shuttle import ResizedFrame, TensorInfo, TrackNetV3Tracker
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
    frames = [_resized_frame(i + 1, value=10 * (i + 1)) for i in range(8)]
    frames[0].image_rgb[0, 0] = [0, 64, 128]
    frames[-1].image_rgb[0, 0] = [24, 48, 96]

    tensor = tracker._input_tensor(frames, background)

    assert tensor.shape == (1, 27, 288, 512)
    np.testing.assert_allclose(tensor[0, :3, 0, 0], [1.0, 128 / 255, 0.0])
    np.testing.assert_allclose(tensor[0, 3:6, 0, 0], [0.0, 64 / 255, 128 / 255])
    np.testing.assert_allclose(tensor[0, 24:27, 0, 0], [24 / 255, 48 / 255, 96 / 255])


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
