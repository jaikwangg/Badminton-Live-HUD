"""SRT source using OpenCV/GStreamer with an FFmpeg raw-frame fallback."""

from __future__ import annotations

import subprocess
from typing import Any

from hud.bus.messages import Frame
from hud.sources.base import Source, to_frame


class SRTSource(Source):
    def __init__(self, url: str, *, width: int = 1280, height: int = 720) -> None:
        self.url = url
        self.width = width
        self.height = height
        self._capture: Any = None
        self._ffmpeg: subprocess.Popen[bytes] | None = None
        self._frame_id = 0

    def open(self) -> None:
        import cv2

        pipeline = (
            f"srtsrc uri={self.url} ! tsdemux ! h264parse ! avdec_h264 ! "
            "videoconvert ! appsink drop=true max-buffers=1 sync=false"
        )
        capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if capture.isOpened():
            self._capture = capture
            return
        capture.release()
        self._ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                self.url,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self.width}x{self.height}",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if self._ffmpeg.stdout is None:
            raise OSError("FFmpeg did not provide a video stream")

    def read(self) -> Frame | None:
        if self._capture is not None:
            ok, image = self._capture.read()
            if not ok:
                return None
            self._frame_id += 1
            return to_frame(image, self._frame_id)
        if self._ffmpeg is None or self._ffmpeg.stdout is None:
            raise RuntimeError("SRT source is not open")
        expected = self.width * self.height * 3
        payload = bytearray()
        while len(payload) < expected:
            chunk = self._ffmpeg.stdout.read(expected - len(payload))
            if not chunk:
                return None
            payload.extend(chunk)
        self._frame_id += 1
        from time import monotonic_ns

        return Frame(self._frame_id, monotonic_ns(), self.width, self.height, bytes(payload))

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._ffmpeg is not None:
            self._ffmpeg.terminate()
            self._ffmpeg.wait(timeout=2)
            self._ffmpeg = None
