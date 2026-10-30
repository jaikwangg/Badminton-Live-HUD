from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from hud.bus import shm_ring
from hud.bus.messages import Frame
from hud.bus.shm_ring import SharedFrameRing
from hud.config import AppConfig
from hud.fusion.state import validate_game_state
from hud.recorder import writer
from hud.recorder.writer import Recorder, VideoEncoder, empty_game_state, read_new_frames


class FakeEncoder:
    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.closed = False

    def write(self, frame: bytes) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True


def _config(recording_dir: Path) -> AppConfig:
    config = AppConfig()
    config.recording = {
        "directory": str(recording_dir),
        "min_free_gb": 0,
        "segment_seconds": 1800,
        "disk_check_interval_s": 5,
        "disk_retry_interval_s": 0.001,
        "state_queue_size": 32,
    }
    return config


def _frame(frame_id: int, ts: int) -> Frame:
    return Frame(frame_id=frame_id, ts=ts, width=2, height=1, image=bytes([frame_id] * 6))


def test_jsonl_matches_video_frames_and_records_real_gap_fills(tmp_path: Path) -> None:
    recorder = Recorder(_config(tmp_path))
    recorder._create_session_directory()
    encoder = FakeEncoder()
    recorder._encoder = encoder  # type: ignore[assignment]

    recorder._record_frame(_frame(10, 1_000), [])
    recorder._record_frame(_frame(13, 4_000), [])

    recorder._events.close()
    rows = [
        json.loads(line) for line in (recorder.output_dir / "events.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(encoder.frames) == 4
    assert all("frame_id" in row and "ts" in row for row in rows)
    assert [row.get("missing_frame_id") for row in rows if row["repeat_fill"]] == [11, 12]
    assert rows[-1]["frame_id"] == 13
    assert rows[-1]["ts"] == 4_000
    assert rows[-1]["dropped_before"] == 2
    assert recorder._dropped_frames == 2


def test_each_new_ws_state_and_event_is_written_once(tmp_path: Path) -> None:
    recorder = Recorder(_config(tmp_path))
    recorder._create_session_directory()
    encoder = FakeEncoder()
    recorder._encoder = encoder  # type: ignore[assignment]
    messages = [
        {"v": 1, "ts": 11, "events": [{"ts": 11, "kind": "shot", "data": {"id": 1}}]},
        {"v": 1, "ts": 12, "events": [{"ts": 12, "kind": "shot", "data": {"id": 2}}]},
    ]
    for message in messages:
        recorder._state_messages.put_nowait(message)
    recorder._record_frame(_frame(1, 10), recorder._take_state_updates())
    recorder._record_frame(_frame(2, 20), recorder._take_state_updates())
    recorder._events.close()

    rows = [
        json.loads(line) for line in (recorder.output_dir / "events.jsonl").read_text().splitlines()
    ]
    assert [row["state_updates"] for row in rows] == [
        [{"v": 1, "ts": 11}, {"v": 1, "ts": 12}],
        [],
    ]
    written_events = [event for row in rows for event in row["events"]]
    assert written_events == [event for message in messages for event in message["events"]]
    assert len(encoder.frames) == len(rows) == 2


def test_non_finite_ws_state_is_written_as_null(tmp_path: Path) -> None:
    recorder = Recorder(_config(tmp_path))
    recorder._create_session_directory()
    encoder = FakeEncoder()
    recorder._encoder = encoder  # type: ignore[assignment]
    recorder._record_frame(_frame(1, 10), [{"v": 1, "health": {"fps": float("nan")}}])
    recorder._events.close()

    row = json.loads((recorder.output_dir / "events.jsonl").read_text(encoding="utf-8"))
    assert row["state_updates"] == [{"v": 1, "health": {"fps": None}}]


def test_disk_full_reports_degraded_heartbeat_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    statuses: list[tuple[str, str | None]] = []
    heartbeats: list[dict[str, object]] = []
    recorder = Recorder(
        _config(tmp_path),
        status_callback=lambda status, reason: statuses.append((status, reason)),
        heartbeat_callback=lambda payload: heartbeats.append(json.loads(payload)),
    )
    recorder.config.recording["min_free_gb"] = 1
    free_values = iter((0, 0, 10**12))
    monkeypatch.setattr(
        writer.shutil, "disk_usage", lambda _path: SimpleNamespace(free=next(free_values))
    )
    recorder._last_heartbeat_at -= 2

    asyncio.run(recorder._wait_for_disk_space())

    assert (
        "degraded",
        "free disk space is below the configured minimum; waiting for space",
    ) in statuses
    assert statuses[-1] == ("ok", None)
    assert any(heartbeat["status"] == "degraded" for heartbeat in heartbeats)
    assert recorder.output_dir is not None


def test_segment_muxer_uses_recoverable_mkv_and_configured_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stderr = io.BytesIO()

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    calls: list[list[str]] = []

    def fake_popen(command: list[str], **_kwargs: object) -> FakeProcess:
        calls.append(command)
        return FakeProcess()

    monkeypatch.setattr(VideoEncoder, "_probe_nvenc", lambda _self: True)
    monkeypatch.setattr(writer.subprocess, "Popen", fake_popen)
    encoder = VideoEncoder(
        "ffmpeg",
        1280,
        720,
        60,
        tmp_path / "video_%03d.mkv",
        segment_seconds=120,
    )
    encoder.close()

    command = calls[0]
    assert command[command.index("-f") + 1] == "rawvideo"
    segment_index = command.index("-segment_time")
    assert command[segment_index + 1] == "120"
    assert command[command.index("-segment_format") + 1] == "matroska"
    assert command[-1].endswith("video_%03d.mkv")
    assert "-r" not in command


def test_snapshot_is_merged_and_empty_state_uses_current_schema(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    override.write_text("recording:\n  min_free_gb: 1\n", encoding="utf-8")
    recorder = Recorder(_config(tmp_path / "recordings"), config_snapshot=override)
    output = recorder._create_session_directory()
    snapshot = yaml.safe_load((output / "config.snapshot.yaml").read_text(encoding="utf-8"))
    state = empty_game_state(123)

    assert snapshot["source"]["width"] == 1280
    assert snapshot["recording"]["segment_seconds"] == 1800
    assert snapshot["recording"]["min_free_gb"] == 0
    assert state["shuttle"]["landing_xy_m_est"] is None
    assert "court_xy_m" not in state["shuttle"]
    assert state["health"]["fps"] is None
    assert state["ts"] == 123


@pytest.mark.parametrize(("mode", "count"), [("singles", 2), ("doubles", 4)])
def test_empty_game_state_is_contract_valid(mode: str, count: int) -> None:
    state = empty_game_state(5, mode)

    validate_game_state(state)
    assert state["match"]["mode"] == mode
    assert [player["id"] for player in state["players"]] == list(range(count))
    assert all(
        player["tracked"] is False and player["img_xy"] is None and player["court_xy_m"] is None
        for player in state["players"]
    )
    with pytest.raises(ValueError):
        empty_game_state(5, "mixed")


def test_per_slot_read_never_skips_when_frames_outpace_the_poll() -> None:
    ring = SharedFrameRing(f"hud_test_{uuid.uuid4().hex}", 2, 1)
    try:
        image = np.zeros((1, 2, 3), dtype=np.uint8)
        assert read_new_frames(ring, None) == []
        ring.write(image, 1)
        ring.write(image, 2)
        assert [frame.frame_id for frame in read_new_frames(ring, None)] == [2]  # newest only
        seen = [2]
        for burst in (3, 4, 1, 4):  # up to a full ring between polls
            for _ in range(burst):
                ring.write(image, time.monotonic_ns())
            seen.extend(frame.frame_id for frame in read_new_frames(ring, seen[-1]))
        assert seen == list(range(2, 15))

        # A slot mid-write (odd seq) is skipped rather than read torn.
        ring.write(np.zeros((1, 2, 3), dtype=np.uint8), 0)
        slot = 15 % ring.slots
        header = list(shm_ring._HEADER.unpack_from(ring.shm.buf, slot * ring.slot_size))
        header[4] += 1
        shm_ring._HEADER.pack_into(ring.shm.buf, slot * ring.slot_size, *header)
        assert read_new_frames(ring, 14) == []
    finally:
        ring.close()
        ring.unlink()


def test_heartbeat_loop_does_not_repeat_a_just_sent_heartbeat(tmp_path: Path) -> None:
    beats: list[str] = []
    recorder = Recorder(_config(tmp_path), heartbeat_callback=beats.append)

    async def run() -> None:
        recorder._send_heartbeat_if_due(force=True)  # e.g. a status-change heartbeat
        task = asyncio.create_task(recorder._heartbeat_loop())
        await asyncio.sleep(0.2)
        recorder.stop()
        await task

    asyncio.run(run())
    assert len(beats) == 1


def test_frame_gap_warnings_are_rate_limited(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = Recorder(_config(tmp_path))
    recorder._create_session_directory()
    recorder._encoder = FakeEncoder()  # type: ignore[assignment]

    with caplog.at_level("WARNING", logger=writer.LOGGER.name):
        for frame_id in range(1, 200, 2):  # every other frame missing: 99 gaps
            recorder._record_frame(_frame(frame_id, frame_id), [])
        recorder._last_gap_log_at -= recorder._gap_log_interval_s
        recorder._record_frame(_frame(203, 203), [])
    recorder._events.close()

    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 2
    assert "skipped 1 frame(s) in 1 gap(s)" in warnings[0]
    assert "skipped 101 frame(s) in 99 gap(s)" in warnings[1]
    assert recorder._dropped_frames == 102


def test_reader_catches_timeout_value_error_and_other_exceptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = Recorder(_config(tmp_path))
    attempts = 0

    class FailingConnection:
        async def __aenter__(self) -> FailingConnection:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise subprocess.TimeoutExpired("websocket", 1)
            if attempts == 2:
                raise ValueError("synthetic value error")
            raise RuntimeError("synthetic unexpected error")

        async def __aexit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(writer.websockets, "connect", lambda *_args, **_kwargs: FailingConnection())

    async def stop_after_retry(_delay: float) -> None:
        if attempts >= 3:
            recorder.stop()

    monkeypatch.setattr(writer.asyncio, "sleep", stop_after_retry)

    async def run_reader() -> None:
        await asyncio.wait_for(recorder._read_game_states("ws://example.invalid"), timeout=1)

    asyncio.run(run_reader())
    assert attempts == 3


def test_stop_signal_finalizes_ffmpeg_segment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SIGTERM (or stop() where it cannot be caught) must close FFmpeg's stdin so it exits 0."""
    marker = tmp_path / "ffmpeg.bytes"
    fake_ffmpeg = (
        f"import sys; n = len(sys.stdin.buffer.read()); open({str(marker)!r}, 'w').write(str(n))"
    )
    real_popen = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []

    def spawn_fake(_command: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        processes.append(real_popen([sys.executable, "-c", fake_ffmpeg], **kwargs))  # type: ignore[call-overload]
        return processes[-1]

    monkeypatch.setattr(VideoEncoder, "_probe_nvenc", lambda _self: False)
    monkeypatch.setattr(writer.subprocess, "Popen", spawn_fake)
    monkeypatch.setattr(writer.shutil, "which", lambda _name: sys.executable)
    config = _config(tmp_path / "rec")
    config.bus.name = f"hud_test_{uuid.uuid4().hex}"
    config.source.width, config.source.height = 2, 1
    config.server["ws_port"] = 9  # nothing listens; the state reader just retries
    ring = SharedFrameRing(config.bus.name, 2, 1)
    recorder = Recorder(config, heartbeat_callback=lambda _payload: None)

    async def run() -> None:
        task = asyncio.create_task(recorder.run())
        writer.install_stop_signals(recorder)
        deadline = time.monotonic() + 10
        for count in range(1, 4):  # one at a time: the first poll keeps only the newest frame
            ring.write(np.zeros((1, 2, 3), dtype=np.uint8), time.monotonic_ns())
            while recorder._frame_count < count and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
        if sys.platform == "win32":
            recorder.stop()  # TerminateProcess cannot be caught on Windows
        else:
            os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(task, timeout=10)

    try:
        asyncio.run(run())
    finally:
        ring.close()
        ring.unlink()
    assert processes[0].returncode == 0
    assert marker.read_text() == str(3 * 6)
