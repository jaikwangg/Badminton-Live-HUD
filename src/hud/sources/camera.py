"""USB/V4L2 camera source using OpenCV."""

from __future__ import annotations

from typing import Any

from hud.bus.messages import Frame
from hud.sources.base import Source, to_frame


class CameraSource(Source):
    def __init__(self, device: int = 0, *, width: int = 1280, height: int = 720) -> None:
        self.device = device
        self.width = width
        self.height = height
        self._capture: Any = None
        self._frame_id = 0

    def open(self) -> None:
        import cv2

        self._capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise OSError(f"Unable to open camera device: {self.device}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> Frame | None:
        if self._capture is None:
            raise RuntimeError("Camera source is not open")
        ok, image = self._capture.read()
        if not ok:
            return None
        self._frame_id += 1
        return to_frame(image, self._frame_id)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
