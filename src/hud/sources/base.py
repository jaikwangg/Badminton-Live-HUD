"""Shared capture source interface and frame conversion helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import monotonic_ns
from typing import Any

from hud.bus.messages import Frame


class Source(ABC):
    """A source whose reads return captured BGR frames with monotonic timestamps."""

    @abstractmethod
    def open(self) -> None:
        """Open the underlying source or raise an error."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Read one frame, returning None when no frame is currently available."""

    @abstractmethod
    def close(self) -> None:
        """Release source resources."""


def to_frame(image: Any, frame_id: int) -> Frame:
    """Copy an OpenCV BGR image into the bus frame representation."""
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("Source frames must be BGR images with three channels")
    return Frame(frame_id, monotonic_ns(), width, height, image.tobytes(order="C"))
