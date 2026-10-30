"""Four-slot shared-memory frame ring using per-slot seqlock counters."""

from __future__ import annotations

import os
import struct
import sys
from multiprocessing import resource_tracker, shared_memory
from typing import Any

from hud.bus.messages import Frame

_HEADER = struct.Struct("<QqHHQ")  # frame_id, capture ts, width, height, seqlock seq
_HEADER_SIZE = _HEADER.size
_HAS_TRACK_ARG = sys.version_info >= (3, 13)
_POSIX = os.name == "posix"


def _attach_untracked(name: str) -> shared_memory.SharedMemory:
    """Attach to an existing segment so this process exiting never unlinks it.

    Before 3.13, POSIX attaches register with resource_tracker, which unlinks the segment
    when the attaching process exits, killing the bus for every other worker.
    """
    if _HAS_TRACK_ARG:
        return shared_memory.SharedMemory(name=name, track=False)
    shm = shared_memory.SharedMemory(name=name)
    if _POSIX:
        # ponytail: the tracker keeps one entry per name per process, so a reader in the
        # creator's own process also drops the creator's crash cleanup; fine outside tests.
        resource_tracker.unregister(shm._name, "shared_memory")
    return shm


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
        # The creator owns the segment (and on Windows must keep its handle open, since the
        # segment vanishes once the last handle closes); readers attach without ownership.
        if create:
            self.shm = shared_memory.SharedMemory(
                name=name, create=True, size=self.slot_size * slots
            )
        else:
            self.shm = _attach_untracked(name)

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
