"""Four-slot shared-memory frame ring using per-slot seqlock counters."""

from __future__ import annotations

import struct
from multiprocessing import shared_memory
from typing import Any

from hud.bus.messages import Frame

_HEADER = struct.Struct("<QqHHQ")  # frame_id, capture ts, width, height, seqlock seq
_HEADER_SIZE = _HEADER.size


class SharedFrameRing:
    """Fixed-size BGR frame ring; readers always select the newest complete frame."""

    def __init__(
        self,
        name: str = "hud_frames",
        width: int = 1280,
        height: int = 720,
        slots: int = 4,
        *,
        create: bool = True,
    ) -> None:
        if slots != 4:
            raise ValueError("The frame bus contract requires exactly four slots")
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive")
        self.width = width
        self.height = height
        self.slots = slots
        self.slot_size = _HEADER_SIZE + width * height * 3
        self.shm = shared_memory.SharedMemory(
            name=name, create=create, size=self.slot_size * slots if create else 0
        )

    def write(self, frame: Any, ts: int) -> int:
        """Write a uint8 BGR image and capture timestamp; return its frame id."""
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) != 3 or shape != (self.height, self.width, 3):
            raise ValueError(f"Expected image shape {(self.height, self.width, 3)}")
        image = frame.tobytes(order="C")
        frame_id = self._latest_frame_id() + 1
        slot = frame_id % self.slots
        offset = slot * self.slot_size
        previous_seq = _HEADER.unpack_from(self.shm.buf, offset)[4]
        odd_seq = previous_seq + 1 if previous_seq % 2 == 0 else previous_seq + 2
        _HEADER.pack_into(self.shm.buf, offset, frame_id, ts, self.width, self.height, odd_seq)
        start = offset + _HEADER_SIZE
        self.shm.buf[start : start + len(image)] = image
        _HEADER.pack_into(self.shm.buf, offset, frame_id, ts, self.width, self.height, odd_seq + 1)
        return frame_id

    def _latest_frame_id(self) -> int:
        newest = 0
        for index in range(self.slots):
            values = _HEADER.unpack_from(self.shm.buf, index * self.slot_size)
            if values[4] % 2 == 0:
                newest = max(newest, values[0])
        return newest

    def read_latest(self, retries: int = 3) -> Frame | None:
        """Read the latest stable image, dropping retries when a writer is active."""
        for _ in range(retries):
            candidates: list[tuple[int, int, int, int, int]] = []
            for index in range(self.slots):
                offset = index * self.slot_size
                header = _HEADER.unpack_from(self.shm.buf, offset)
                if header[0] and header[4] % 2 == 0:
                    candidates.append((*header, index))
            if not candidates:
                return None
            frame_id, ts, width, height, seq, slot = max(candidates, key=lambda item: item[0])
            offset = slot * self.slot_size
            start = offset + _HEADER_SIZE
            image = bytes(self.shm.buf[start : start + width * height * 3])
            check = _HEADER.unpack_from(self.shm.buf, offset)
            if check[4] == seq and seq % 2 == 0 and check[0] == frame_id:
                return Frame(frame_id, ts, width, height, image)
        return None

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()

    def __enter__(self) -> SharedFrameRing:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
