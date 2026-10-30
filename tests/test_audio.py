from __future__ import annotations

import json
import subprocess
import sys
import time
import types
import wave
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest

from hud.bus.messages import HitMessage
from hud.config import AppConfig
from hud.workers import audio
from hud.workers.audio import (
    AudioHitDetector,
    AudioWorker,
    has_audio_input_device,
)

try:
    import sounddevice  # noqa: F401
except (ImportError, OSError):  # not installed, or PortAudio missing: stub it so patch() works
    sys.modules["sounddevice"] = types.SimpleNamespace(query_devices=None, InputStream=None)


class FakePublisher:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, message: str) -> None:
        self.messages.append(message)

    def close(self) -> None:
        pass


def _generate_tone(
    freq_hz: float, duration_s: float, sample_rate: int = 44100, amplitude: float = 0.5
) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_detector_identifies_badminton_hit_frequency() -> None:
    detector = AudioHitDetector(sample_rate=44100, low_freq=2000, high_freq=8000, sensitivity=3.0)

    # 1. Background calm noise - should adapt without triggering
    calm = np.random.uniform(-0.005, 0.005, size=1024).astype(np.float32)
    hit_calm = detector.process_chunk(calm, timestamp_ns=1_000_000_000)
    assert hit_calm is None

    # 2. Badminton racket string hit (4000 Hz sharp ping)
    ping = _generate_tone(freq_hz=4000, duration_s=1024 / 44100, sample_rate=44100, amplitude=0.4)
    hit = detector.process_chunk(ping, timestamp_ns=1_050_000_000)

    assert hit is not None
    assert hit.v == 1
    assert hit.ts == 1_050_000_000
    assert hit.strength is not None
    assert 0.1 <= hit.strength <= 1.0


def test_detector_filters_out_low_frequency_noise() -> None:
    detector = AudioHitDetector(sample_rate=44100, low_freq=2000, high_freq=8000, sensitivity=3.0)

    # Low frequency rumble (e.g. 100 Hz footstep) with high amplitude
    rumble = _generate_tone(freq_hz=100, duration_s=1024 / 44100, sample_rate=44100, amplitude=0.8)
    hit = detector.process_chunk(rumble, timestamp_ns=1_000_000_000)

    # Bandpass filter (2-8 kHz) removes 100 Hz rumble, so no hit is triggered
    assert hit is None


def test_detector_refractory_lockout_150ms() -> None:
    detector = AudioHitDetector(
        sample_rate=44100,
        low_freq=2000,
        high_freq=8000,
        refractory_ms=150.0,
        sensitivity=2.5,
    )
    ping = _generate_tone(freq_hz=4000, duration_s=1024 / 44100, sample_rate=44100, amplitude=0.5)

    # First hit at t = 100 ms
    t1 = 100_000_000
    hit1 = detector.process_chunk(ping, timestamp_ns=t1)
    assert hit1 is not None

    # Echo / retrigger at t = 160 ms (60 ms later, < 150 ms lockout)
    t2 = 160_000_000
    hit2 = detector.process_chunk(ping, timestamp_ns=t2)
    assert hit2 is None

    # Next legitimate hit at t = 260 ms (160 ms later, > 150 ms lockout)
    t3 = 260_000_000
    hit3 = detector.process_chunk(ping, timestamp_ns=t3)
    assert hit3 is not None
    assert hit3.ts == t3


def test_worker_publishes_hit_with_time_offset_and_heartbeat() -> None:
    detector = AudioHitDetector(sample_rate=44100, low_freq=2000, high_freq=8000, sensitivity=2.0)
    publisher = FakePublisher()
    # 25 ms offset
    worker = AudioWorker(detector, publisher, offset_ms=25.0)

    ping = _generate_tone(freq_hz=3500, duration_s=1024 / 44100, sample_rate=44100, amplitude=0.5)
    t_base = 500_000_000
    hit = worker.process(ping, timestamp_ns=t_base)

    assert hit is not None
    # 500ms + 25ms offset = 525ms (525,000,000 ns)
    assert hit.ts == 525_000_000

    # Verify messages published: HitMessage and HeartbeatMessage
    hit_msgs = [
        HitMessage.from_json(m) for m in publisher.messages if json.loads(m).get("type") == "hit"
    ]
    assert len(hit_msgs) == 1
    assert hit_msgs[0].ts == 525_000_000

    heartbeat_msgs = [
        json.loads(m) for m in publisher.messages if json.loads(m).get("type") == "heartbeat"
    ]
    assert len(heartbeat_msgs) >= 1
    assert heartbeat_msgs[0]["worker"] == "audio"
    assert heartbeat_msgs[0]["v"] == 1


def test_has_audio_input_device_when_none_available() -> None:
    with patch("sounddevice.query_devices", return_value=[]):
        assert has_audio_input_device() is False

    with patch(
        "sounddevice.query_devices", return_value=[{"name": "Speaker", "max_input_channels": 0}]
    ):
        assert has_audio_input_device() is False

    with patch(
        "sounddevice.query_devices", return_value=[{"name": "Mic", "max_input_channels": 1}]
    ):
        assert has_audio_input_device() is True


class OneTickStop:
    """Event stand-in whose wait() times out once, then reports stop."""

    def __init__(self) -> None:
        self.calls = 0

    def wait(self, timeout: float) -> bool:
        self.calls += 1
        return self.calls > 1


def test_degraded_heartbeat_goes_to_stdout_only(capsys) -> None:
    audio._degraded_heartbeat(OneTickStop())
    heartbeat = json.loads(capsys.readouterr().out.strip())
    assert heartbeat["v"] == 1
    assert heartbeat["type"] == "heartbeat"
    assert heartbeat["worker"] == "audio"


def _run_degraded(has_device: bool, input_stream_error: Exception | None) -> list[str]:
    publisher = FakePublisher()
    degraded: list[bool] = []
    stream = patch("sounddevice.InputStream", side_effect=input_stream_error)
    with (
        patch.object(audio, "ZmqPublisher", return_value=publisher),
        patch.object(audio, "has_audio_input_device", return_value=has_device),
        patch.object(audio, "_degraded_heartbeat", lambda stop: degraded.append(True)),
        stream,
    ):
        audio.run(AppConfig())
    assert degraded == [True]
    return publisher.messages


def test_run_without_microphone_degrades_instead_of_crashing() -> None:
    assert _run_degraded(has_device=False, input_stream_error=None) == []


def test_run_survives_failed_audio_device() -> None:
    assert _run_degraded(has_device=True, input_stream_error=OSError("device busy")) == []


def test_capture_publishes_hits_from_main_thread_then_returns_when_stream_stops() -> None:
    ping = _generate_tone(freq_hz=4000, duration_s=1024 / 44100, amplitude=0.5)

    class FakeStream:
        def __init__(self, callback, **kwargs) -> None:
            self.callback = callback
            self.checks = 0

        def __enter__(self) -> FakeStream:
            self.callback(ping[:, None], len(ping), None, None)
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        @property
        def active(self) -> bool:
            self.checks += 1
            return self.checks <= 2

    publisher = FakePublisher()
    worker = AudioWorker(AudioHitDetector(sensitivity=2.0), publisher, offset_ms=10.0)
    with patch("sounddevice.InputStream", FakeStream):
        audio._capture(worker, 44100, 1024, 8, Event())
    hits = [HitMessage.from_json(m) for m in publisher.messages if '"hit"' in m]
    assert len(hits) == 1


FFMPEG = audio.find_ffmpeg(AppConfig())
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available")


def _write_click_wav(path, duration_s: float, click_times_s: list[float], rate: int = 16000):
    """Quiet noise with 20 ms 4 kHz clicks, written with the stdlib wave module."""
    rng = np.random.default_rng(0)
    signal = rng.uniform(-0.003, 0.003, int(duration_s * rate))
    click = _generate_tone(4000, 0.02, sample_rate=rate, amplitude=0.6)
    for t in click_times_s:
        start = int(t * rate)
        signal[start : start + len(click)] += click
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())


class StopAfterHits(FakePublisher):
    def __init__(self, stop: Event, hits: int) -> None:
        super().__init__()
        self.stop, self.hits = stop, hits

    def send(self, message: str) -> None:
        super().send(message)
        if sum('"hit"' in m for m in self.messages) >= self.hits:
            self.stop.set()


def _hit_times(messages: list[str]) -> list[int]:
    return [HitMessage.from_json(m).ts for m in messages if '"hit"' in m]


@needs_ffmpeg
def test_file_audio_hits_land_on_click_times_in_real_time(tmp_path) -> None:
    clip = tmp_path / "clicks.wav"
    _write_click_wav(clip, 1.0, [0.3, 0.8])
    publisher = FakePublisher()
    worker = AudioWorker(AudioHitDetector(sample_rate=16000, sensitivity=3.0), publisher)
    started = time.monotonic_ns()
    audio._replay_file(worker, FFMPEG, str(clip), False, 16000, 256, Event())
    elapsed_s = (time.monotonic_ns() - started) / 1e9
    hits = _hit_times(publisher.messages)
    assert len(hits) == 2
    # Chunks are 16 ms, so a hit is stamped at most one chunk before its click.
    assert abs((hits[1] - hits[0]) / 1e6 - 500) <= 20
    assert 0.25 <= (hits[0] - started) / 1e9 <= 0.8  # click at 0.3 s plus ffmpeg start-up
    assert elapsed_s >= 0.95  # paced in real time, not decoded as fast as possible


@needs_ffmpeg
def test_file_audio_loop_keeps_one_continuous_clock(tmp_path) -> None:
    clip = tmp_path / "loop.wav"
    _write_click_wav(clip, 0.5, [0.2])
    stop = Event()
    publisher = StopAfterHits(stop, 3)
    worker = AudioWorker(AudioHitDetector(sample_rate=16000, sensitivity=3.0), publisher)
    audio._replay_file(worker, FFMPEG, str(clip), True, 16000, 256, stop)
    hits = _hit_times(publisher.messages)
    assert len(hits) == 3
    assert all(abs((b - a) / 1e6 - 500) <= 20 for a, b in zip(hits, hits[1:], strict=False))


def _run_file_source(path: str) -> list[str]:
    config = AppConfig()
    config.source.path = path
    config.workers["audio"] = {"sample_rate": 16000}
    publisher = FakePublisher()
    degraded: list[bool] = []
    with (
        patch.object(audio, "ZmqPublisher", return_value=publisher),
        patch.object(audio, "has_audio_input_device", side_effect=AssertionError("mic used")),
        patch.object(audio, "_degraded_heartbeat", lambda stop: degraded.append(True)),
    ):
        audio.run(config)
    assert degraded == [True]
    return publisher.messages


@needs_ffmpeg
def test_file_without_audio_track_degrades(tmp_path) -> None:
    clip = tmp_path / "silent.mp4"
    subprocess.run(
        [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.3"]
        + ["-c:v", "mpeg4", str(clip)],
        check=True,
    )
    assert _run_file_source(str(clip)) == []


def test_file_source_without_ffmpeg_degrades() -> None:
    with patch.object(audio, "find_ffmpeg", return_value=None):
        assert _run_file_source("match.mp4") == []
