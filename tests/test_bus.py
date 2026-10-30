from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid

import pytest

from hud.bus import shm_ring
from hud.bus.messages import (
    HeartbeatMessage,
    HitMessage,
    PlayersMessage,
    ShuttleMessage,
    message_from_json,
)
from hud.bus.shm_ring import SharedFrameRing


class FakeImage:
    def __init__(self, width: int, height: int, value: int) -> None:
        self.shape = (height, width, 3)
        self.value = value

    def tobytes(self, order: str = "C") -> bytes:
        assert order == "C"
        return bytes([self.value]) * (self.shape[0] * self.shape[1] * self.shape[2])


def test_all_worker_messages_round_trip() -> None:
    messages = [
        PlayersMessage(
            frame_id=3,
            ts=8,
            players=[
                {"track_id": 1, "bbox": [1, 2, 3, 4], "foot_px": [2, 4], "conf": 0.9, "app": None}
            ],
        ),
        ShuttleMessage(frame_id=3, ts=8, visible=True, px=[4, 5], conf=0.8),
        HitMessage(ts=8, strength=0.72),
        HeartbeatMessage(worker="players", ts=8, fps=59.8, proc_ms=6.1),
    ]
    for message in messages:
        decoded = message_from_json(message.to_json())
        assert decoded.to_json() == message.to_json()


def test_players_app_round_trips_and_defaults_to_null() -> None:
    with_app = PlayersMessage(
        players=[
            {"track_id": 1, "bbox": [1, 2, 3, 4], "foot_px": [2, 4], "conf": 0.9, "app": [0.5, 0.5]}
        ]
    )
    assert PlayersMessage.from_json(with_app.to_json()).players[0].app == [0.5, 0.5]
    old = (
        '{"v":1,"type":"players","frame_id":1,"ts":2,'
        '"players":[{"track_id":1,"bbox":[1,2,3,4],"foot_px":[2,4],"conf":0.9}]}'
    )
    decoded = PlayersMessage.from_json(old)
    assert decoded.players[0].app is None
    assert '"app":null' in decoded.to_json()


def test_ring_readers_never_observe_torn_frames() -> None:
    name = f"hud_test_{uuid.uuid4().hex}"
    width, height = 96, 54
    writer = SharedFrameRing(name, width, height)
    readers = [SharedFrameRing(name, width, height, create=False) for _ in range(2)]
    stop = threading.Event()
    failures: list[str] = []

    def write_frames() -> None:
        index = 1
        while not stop.is_set():
            writer.write(FakeImage(width, height, index % 251), index)
            index += 1

    def read_frames(ring: SharedFrameRing) -> None:
        while not stop.is_set():
            frame = ring.read_latest()
            if frame is None:
                continue
            if frame.ts != frame.frame_id or any(byte != frame.image[0] for byte in frame.image):
                failures.append("reader observed inconsistent frame data")
                stop.set()

    threads = [threading.Thread(target=write_frames)] + [
        threading.Thread(target=read_frames, args=(ring,)) for ring in readers
    ]
    try:
        for thread in threads:
            thread.start()
        time.sleep(10)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=2)
        for ring in readers:
            ring.close()
        writer.close()
        writer.unlink()
    assert not failures


def test_reader_attach_untracks_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    name = f"hud_test_{uuid.uuid4().hex}"
    writer = SharedFrameRing(name, 4, 2)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(shm_ring, "_HAS_TRACK_ARG", False)
    monkeypatch.setattr(shm_ring, "_POSIX", True)
    monkeypatch.setattr(
        shm_ring.resource_tracker, "unregister", lambda n, rtype: calls.append((n, rtype))
    )
    reader = SharedFrameRing(name, 4, 2, create=False)
    try:
        assert calls == [(reader.shm._name, "shared_memory")]
    finally:
        reader.close()
        writer.close()
        writer.unlink()


@pytest.mark.skipif(
    os.name != "posix", reason="Windows has no unlink; a segment lives while any handle is open"
)
def test_reader_process_exit_keeps_segment() -> None:
    name = f"hud_test_{uuid.uuid4().hex}"
    writer = SharedFrameRing(name, 4, 2)
    try:
        code = (
            "from hud.bus.shm_ring import SharedFrameRing; "
            f"SharedFrameRing({name!r}, 4, 2, create=False).close()"
        )
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
        # capture_output waits for pipe EOF, i.e. until the child's resource_tracker exits too.
        subprocess.run(
            [sys.executable, "-c", code], env=env, check=True, timeout=30, capture_output=True
        )
        SharedFrameRing(name, 4, 2, create=False).close()  # raises if the reader unlinked it
    finally:
        writer.close()
        writer.unlink()
