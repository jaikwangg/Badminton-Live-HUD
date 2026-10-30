"""Audio hit worker with 2-8 kHz band-pass onset detection and adaptive noise floor."""

from __future__ import annotations

import json
import logging
import time
from threading import Event
from typing import Any, Protocol

import numpy as np

from hud.bus.messages import HeartbeatMessage, HitMessage
from hud.config import AppConfig, load_config

logger = logging.getLogger(__name__)


class Publisher(Protocol):
    def send(self, message: str) -> None: ...

    def close(self) -> None: ...


class ZmqPublisher:
    def __init__(self, endpoint: str) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("Install pyzmq to publish worker messages") from exc
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 4)
        self._socket.bind(endpoint)

    def send(self, message: str) -> None:
        self._socket.send_string(message)

    def close(self) -> None:
        self._socket.close(linger=0)


class AudioHitDetector:
    """Detect badminton racket hits using band-pass filtering and adaptive thresholding."""

    def __init__(
        self,
        sample_rate: int = 44100,
        low_freq: float = 2000.0,
        high_freq: float = 8000.0,
        refractory_ms: float = 150.0,
        sensitivity: float = 3.5,
        min_threshold: float = 0.015,
        alpha: float = 0.95,
    ) -> None:
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        self.high_freq = min(high_freq, sample_rate * 0.45)
        self.refractory_ns = int(refractory_ms * 1_000_000)
        self.sensitivity = sensitivity
        self.min_threshold = min_threshold
        self.alpha = alpha

        self.noise_floor = min_threshold
        self.last_hit_ns = -self.refractory_ns
        self._sos = self._build_bandpass()

    def _build_bandpass(self) -> np.ndarray:
        try:
            from scipy.signal import butter

            return butter(
                4,
                [self.low_freq, self.high_freq],
                btype="bandpass",
                fs=self.sample_rate,
                output="sos",
            )
        except ImportError:
            # Fallback identity when scipy is absent
            return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    def filter_audio(self, audio: np.ndarray) -> np.ndarray:
        try:
            from scipy.signal import sosfilt

            return sosfilt(self._sos, audio)
        except ImportError:
            return audio

    def process_chunk(self, chunk: np.ndarray, timestamp_ns: int) -> HitMessage | None:
        """Process an audio buffer and return a HitMessage if an onset exceeds threshold."""
        if chunk.dtype != np.float32 and chunk.dtype != np.float64:
            chunk = chunk.astype(np.float32) / 32768.0

        if chunk.ndim > 1:
            chunk = chunk.mean(axis=-1)

        filtered = self.filter_audio(chunk)
        rms = float(np.sqrt(np.mean(filtered**2)))

        threshold = max(self.noise_floor * self.sensitivity, self.min_threshold)

        # Update noise floor adaptively using EMA on non-hit / calm audio
        if rms < threshold:
            self.noise_floor = self.alpha * self.noise_floor + (1.0 - self.alpha) * rms

        # Check for onset hit condition
        if rms >= threshold:
            if timestamp_ns - self.last_hit_ns >= self.refractory_ns:
                self.last_hit_ns = timestamp_ns
                excess = max(0.0, rms - threshold)
                strength = min(1.0, max(0.1, 0.5 + (excess / max(threshold, 1e-4)) * 0.5))
                return HitMessage(ts=timestamp_ns, strength=round(float(strength), 3))

        return None


class AudioWorker:
    """Worker managing audio stream consumption, hit detection, and heartbeat publishing."""

    def __init__(
        self,
        detector: AudioHitDetector,
        publisher: Publisher,
        *,
        offset_ms: float = 0.0,
    ) -> None:
        self.detector = detector
        self.publisher = publisher
        self.offset_ms = offset_ms
        self.offset_ns = int(offset_ms * 1_000_000)

        self.last_heartbeat = 0.0
        self.started = time.monotonic()
        self.processed_chunks = 0
        self.last_proc_ms: float | None = None
        self.last_ts = 0

    def _heartbeat(self) -> None:
        now = time.monotonic()
        fps = self.processed_chunks / max(now - self.started, 1e-9)
        msg = HeartbeatMessage(
            worker="audio",
            ts=self.last_ts,
            fps=fps,
            proc_ms=self.last_proc_ms,
        ).to_json()
        self.publisher.send(msg)
        print(msg, flush=True)
        self.last_heartbeat = now

    def process(
        self,
        chunk: np.ndarray,
        timestamp_ns: int | None = None,
    ) -> HitMessage | None:
        started_ns = time.perf_counter_ns()
        if timestamp_ns is None:
            timestamp_ns = time.monotonic_ns() + self.offset_ns
        else:
            timestamp_ns += self.offset_ns

        self.last_ts = timestamp_ns
        hit = self.detector.process_chunk(chunk, timestamp_ns)

        if hit is not None:
            self.publisher.send(hit.to_json())

        self.last_proc_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        self.processed_chunks += 1

        if time.monotonic() - self.last_heartbeat >= 1.0:
            self._heartbeat()

        return hit


def has_audio_input_device() -> bool:
    """Return True if at least one audio input device is available."""
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        if not devices:
            return False
        if isinstance(devices, dict):
            return int(devices.get("max_input_channels", 0)) > 0
        return any(int(d.get("max_input_channels", 0)) > 0 for d in devices)
    except Exception:
        return False


def run(config: AppConfig) -> None:
    settings: dict[str, Any] = config.workers.get("audio", {})
    sample_rate = int(settings.get("sample_rate", 44100))
    chunk_size = int(settings.get("chunk_size", 1024))
    offset_ms = float(settings.get("offset_ms", 0.0))
    low_freq = float(settings.get("low_freq", 2000.0))
    high_freq = float(settings.get("high_freq", 8000.0))
    refractory_ms = float(settings.get("refractory_ms", 150.0))
    sensitivity = float(settings.get("sensitivity", 3.5))

    detector = AudioHitDetector(
        sample_rate=sample_rate,
        low_freq=low_freq,
        high_freq=high_freq,
        refractory_ms=refractory_ms,
        sensitivity=sensitivity,
    )
    publisher = ZmqPublisher("ipc:///tmp/hud_audio")
    worker = AudioWorker(detector, publisher, offset_ms=offset_ms)

    stop_event = Event()

    if not has_audio_input_device():
        # Degraded state: no microphone available.
        # Emit stdout heartbeat so supervisor does not crash-loop,
        # but omit ZMQ heartbeat so fusion marks status as "down".
        logger.warning("No microphone detected. Audio worker operating in degraded down state.")
        try:
            while not stop_event.wait(1.0):
                # Print supervisor heartbeat to prevent restart thrashing
                print(
                    json.dumps(
                        {
                            "v": 1,
                            "type": "heartbeat",
                            "worker": "audio",
                            "ts": time.monotonic_ns(),
                            "fps": 0.0,
                            "proc_ms": 0.0,
                        }
                    ),
                    flush=True,
                )
        finally:
            publisher.close()
        return

    import sounddevice as sd

    def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            logger.warning("Audio input overflow/underflow: %s", status)
        worker.process(indata.copy())

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            blocksize=chunk_size,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ):
            while not stop_event.wait(0.5):
                if time.monotonic() - worker.last_heartbeat >= 1.0:
                    worker._heartbeat()
    finally:
        publisher.close()


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()
