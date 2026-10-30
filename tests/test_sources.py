from __future__ import annotations

import sys
import types
import uuid

import pytest

from hud.bus.messages import Frame
from hud.bus.shm_ring import SharedFrameRing
from hud.config import AppConfig, BusConfig, SourceConfig
from hud.sources import capture
from hud.sources.base import Source
from hud.sources.capture import CaptureProcess, create_source
from hud.sources.file_replay import FileReplaySource
from hud.sources.srt import SRTSource


class FakeImage:
    shape = (2, 3, 3)

    def tobytes(self, order: str = "C") -> bytes:
        assert order == "C"
        return bytes(range(18))


class FakeCapture:
    def __init__(self, *_: object) -> None:
        self.frames = [FakeImage(), FakeImage()]
        self.set_calls: list[tuple[int, int]] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, _: int) -> float:
        return 60.0

    def read(self) -> tuple[bool, FakeImage | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def set(self, prop: int, value: int) -> None:
        self.set_calls.append((prop, value))
        self.frames = [FakeImage()]

    def release(self) -> None:
        self.released = True


def test_file_replay_paces_frames_and_can_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = FakeCapture()
    cv2 = types.SimpleNamespace(
        VideoCapture=lambda *_: capture,
        CAP_PROP_FPS=5,
        CAP_PROP_POS_FRAMES=1,
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    source = FileReplaySource("match.mp4", loop=True)
    source.open()
    first = source.read()
    second = source.read()
    third = source.read()
    assert first is not None and second is not None and third is not None
    assert first.width == 3 and first.height == 2 and len(first.image) == 18
    assert second.ts - first.ts >= 15_000_000
    assert capture.set_calls == [(cv2.CAP_PROP_POS_FRAMES, 0)]
    source.close()
    assert capture.released


def test_source_type_selects_configured_backend() -> None:
    file_config = AppConfig(source=SourceConfig(type="file", path="match.mp4", loop=True))
    srt_config = AppConfig(source=SourceConfig(type="srt", path="srt://127.0.0.1:9000"))
    assert isinstance(create_source(file_config), FileReplaySource)
    assert isinstance(create_source(srt_config), SRTSource)


def test_source_type_without_required_path_is_rejected() -> None:
    with pytest.raises(ValueError):
        create_source(AppConfig(source=SourceConfig(type="srt")))


def _capture_config() -> AppConfig:
    name = f"hud_test_{uuid.uuid4().hex[:8]}"
    return AppConfig(source=SourceConfig(width=3, height=2), bus=BusConfig(name=name))


def test_capture_resizes_frames_that_do_not_match_the_bus() -> None:
    config = _capture_config()
    seen: list[Frame | None] = []

    class BiggerSource(Source):
        def open(self) -> None:
            pass

        def read(self) -> Frame | None:
            if not seen:
                seen.append(None)
                return Frame(1, 7, 6, 4, bytes(6 * 4 * 3))  # 6x4 into a 3x2 bus
            reader = SharedFrameRing(config.bus.name, 3, 2, create=False)
            seen[0] = reader.read_latest()
            reader.close()
            process.stop()
            return None

        def close(self) -> None:
            pass

    process = CaptureProcess(config, BiggerSource)
    process.run()
    assert seen[0] is not None
    assert (seen[0].width, seen[0].height, seen[0].ts) == (3, 2, 7)


def test_capture_keeps_heartbeating_while_reconnecting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clock = [0.0]
    monkeypatch.setattr(capture.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(capture.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))

    def failing_source() -> Source:
        if clock[0] > 10:
            process.stop()
        raise OSError("camera unplugged")

    process = CaptureProcess(_capture_config(), failing_source)
    process.run()
    beats = [line for line in capsys.readouterr().out.splitlines() if "heartbeat" in line]
    assert len(beats) >= 9  # roughly one per second, never a 3 s gap
