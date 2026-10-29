"""Realtime file replay source."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from hud.bus.messages import Frame
from hud.sources.base import Source, to_frame


class FileReplaySource(Source):
    def __init__(self, path: str | Path, *, loop: bool = False) -> None:
        self.path = str(path)
        self.loop = loop
        self._capture: Any = None
        self._frame_id = 0
        self._fps = 0.0
        self._next_frame_at = 0.0
        self._position_frames = 1

    def open(self) -> None:
        import cv2

        self._capture = cv2.VideoCapture(self.path)
        if not self._capture.isOpened():
            raise OSError(f"Unable to open replay file: {self.path}")
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._position_frames = cv2.CAP_PROP_POS_FRAMES
        if self._fps <= 0:
            self._capture.release()
            self._capture = None
            raise ValueError(f"Replay file has invalid FPS: {self.path}")
        self._next_frame_at = time.monotonic()

    def read(self) -> Frame | None:
        if self._capture is None:
            raise RuntimeError("Replay source is not open")
        delay = self._next_frame_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        ok, image = self._capture.read()
        if not ok and self.loop:
            self._capture.set(self._position_frames, 0)
            ok, image = self._capture.read()
        if not ok:
            return None
        self._frame_id += 1
        self._next_frame_at = max(self._next_frame_at + 1.0 / self._fps, time.monotonic())
        return to_frame(image, self._frame_id)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
