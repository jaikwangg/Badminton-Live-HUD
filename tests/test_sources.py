from __future__ import annotations

import sys
import types

import pytest

from hud.config import AppConfig, SourceConfig
from hud.sources.capture import create_source
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
