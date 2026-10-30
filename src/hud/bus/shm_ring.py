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
_FIELDS = struct.Struct("<QqHH")  # the header minus seq, which is stored on its own
_SEQ_OFFSET = _FIELDS.size
_SEQ = struct.Struct("<Q")
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
        previous_seq = self._seq(offset)
        odd_seq = previous_seq + 1 if previous_seq % 2 == 0 else previous_seq + 2
        # seq goes odd before anything else changes and even only after everything is written.
        self._set_seq(offset, odd_seq)
        _FIELDS.pack_into(self.shm.buf, offset, frame_id, ts, self.width, self.height)
        start = offset + _HEADER_SIZE
        self.shm.buf[start : start + len(image)] = image
        self._set_seq(offset, odd_seq + 1)
        return frame_id

    def _seq(self, offset: int) -> int:
        return _SEQ.unpack_from(self.shm.buf, offset + _SEQ_OFFSET)[0]

    def _set_seq(self, offset: int, seq: int) -> None:
        # Not struct.pack_into: it zero-fills its target first, so readers would briefly see
        # seq 0 (even). That, applied to the whole header, was the torn-header bug.
        start = offset + _SEQ_OFFSET
        self.shm.buf[start : start + _SEQ.size] = seq.to_bytes(_SEQ.size, "little")

    def _read_slot(self, index: int, newer_than: int) -> Frame | None:
        """Copy one slot if it holds a complete frame newer than ``newer_than``, else None."""
        offset = index * self.slot_size
        seq = self._seq(offset)
        if seq % 2:
            return None
        frame_id, ts, width, height = _FIELDS.unpack_from(self.shm.buf, offset)
        if frame_id <= newer_than or (width, height) != (self.width, self.height):
            return None
        start = offset + _HEADER_SIZE
        image = bytes(self.shm.buf[start : start + width * height * 3])
        if self._seq(offset) != seq:
            return None
        return Frame(frame_id, ts, width, height, image)

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
            newest, slot = 0, -1
            for index in range(self.slots):
                offset = index * self.slot_size
                if self._seq(offset) % 2 == 0:
                    frame_id = _FIELDS.unpack_from(self.shm.buf, offset)[0]
                    if frame_id > newest:
                        newest, slot = frame_id, index
            if slot < 0:
                return None
            frame = self._read_slot(slot, 0)
            if frame is not None:
                return frame
        return None

    def read_since(self, frame_id: int) -> list[Frame]:
        """Return every stable frame newer than ``frame_id``, oldest first.

        A slot the writer reuses mid-copy is skipped; it holds a newer frame for the next call.
        """
        frames = [self._read_slot(index, frame_id) for index in range(self.slots)]
        return sorted((f for f in frames if f is not None), key=lambda f: f.frame_id)

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()

    def __enter__(self) -> SharedFrameRing:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
