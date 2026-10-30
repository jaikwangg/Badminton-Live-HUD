"""Independent frame-bus recorder with recoverable FFmpeg segments and JSONL state."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import websockets
import yaml

from hud.bus.messages import Frame, HeartbeatMessage
from hud.bus.shm_ring import SharedFrameRing
from hud.config import AppConfig, load_config

LOGGER = logging.getLogger(__name__)
StatusCallback = Callable[[str, str | None], None]


def empty_game_state(ts: int) -> dict[str, Any]:
    """Return the latest Contract v1 state with explicit unknowns."""
    return {
        "v": 1,
        "ts": ts,
        "match": {"mode": "singles", "score": [None, None], "games": [None, None], "server": None},
        "rally": {"active": False, "n": 0, "t_ms": 0, "shots": 0},
        "players": [
            {
                "id": player_id,
                "name": None,
                "color": None,
                "side": None,
                "tracked": False,
                "img_xy": None,
                "bbox": None,
                "court_xy_m": None,
                "speed_mps": None,
                "dist_m": None,
            }
            for player_id in range(2)
        ],
        "shuttle": {
            "status": "lost",
            "img_xy": None,
            "landing_xy_m_est": None,
            "last_shot": {"by": None, "speed_kmh_est": None, "type": None},
        },
        "events": [],
        "health": {
            "fps": None,
            "latency_ms_p50": None,
            "latency_ms_p99": None,
            "dropped": 0,
            "workers": {"players": "down", "shuttle": "down", "audio": "down"},
        },
    }


class VideoEncoder:
    """Pipe raw frames into FFmpeg's segment muxer using independently readable MKV files."""

    def __init__(
        self,
        ffmpeg: str,
        width: int,
        height: int,
        fps: float,
        output_pattern: Path,
        *,
        segment_seconds: float,
        start_number: int = 0,
    ) -> None:
        self.ffmpeg = ffmpeg
        self.width = width
        self.height = height
        self.fps = fps
        self.output_pattern = output_pattern
        self.segment_seconds = segment_seconds
        self.start_number = start_number
        self.codec = "h264_nvenc" if self._probe_nvenc() else "libx264"
        self.process: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self.start()

    def _probe_nvenc(self) -> bool:
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={self.width}x{self.height}:r={self.fps}:d=0.04",
            "-frames:v",
            "1",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ]
        try:
            result = subprocess.run(command, capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            LOGGER.exception("FFmpeg NVENC probe failed; falling back to libx264")
            return False
        return result.returncode == 0

    def start(self) -> None:
        command = [
            self.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            self.codec,
        ]
        command.extend(
            ["-preset", "p4"]
            if self.codec == "h264_nvenc"
            else ["-preset", "veryfast", "-crf", "23"]
        )
        command.extend(
            [
                "-g",
                str(max(1, round(self.fps * 2))),
                "-f",
                "segment",
                "-segment_time",
                str(self.segment_seconds),
                "-segment_start_number",
                str(self.start_number),
                "-reset_timestamps",
                "1",
                "-segment_format",
                "matroska",
                str(self.output_pattern),
            ]
        )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self.process.stdin is None or self.process.stderr is None:
            raise RuntimeError("FFmpeg did not open raw-frame input and log pipes")
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self.process.stderr,),
            name="recorder-ffmpeg-log",
            daemon=True,
        )
        self._stderr_thread.start()

    @staticmethod
    def _drain_stderr(stream: Any) -> None:
        for raw_line in iter(stream.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                LOGGER.warning("ffmpeg: %s", line)

    def write(self, frame: bytes) -> None:
        if self.process is None or self.process.stdin is None or self.process.poll() is not None:
            raise RuntimeError("Video encoder stopped unexpectedly")
        self.process.stdin.write(frame)

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        finally:
            if process.stderr is not None:
                process.stderr.close()
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=1)
        self.process = None
        if return_code != 0:
            raise RuntimeError(f"FFmpeg exited with status {return_code}")


class Recorder:
    def __init__(
        self,
        config: AppConfig,
        *,
        config_snapshot: str | Path | None = None,
        status_callback: StatusCallback | None = None,
        heartbeat_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        # Keep the argument for CLI compatibility; snapshots are always the merged AppConfig.
        self.config_snapshot = Path(config_snapshot) if config_snapshot else None
        self.status_callback = status_callback or self._log_status
        self.heartbeat_callback = heartbeat_callback or self._stdout_heartbeat
        self.status = "ok"
        self.reason: str | None = None
        self._stopping = asyncio.Event()
        queue_size = int(config.recording.get("state_queue_size", 2048))
        if queue_size < 1:
            raise ValueError("recording.state_queue_size must be positive")
        self._state_messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=queue_size)
        self._last_frame_id: int | None = None
        self._last_frame_ts: int | None = None
        self._last_frame_image: bytes | None = None
        self._frame_count = 0
        self._dropped_frames = 0
        self._segment_start_number = 0
        self._encoder: VideoEncoder | None = None
        self._events: Any = None
        self._last_disk_check = 0.0
        self._started_at = time.monotonic()
        self._last_heartbeat_at = self._started_at
        self._last_ts = 0
        self._last_proc_ms: float | None = None
        self.output_dir: Path | None = None

    @staticmethod
    def _log_status(status: str, reason: str | None) -> None:
        LOGGER.info(
            "recorder status=%s reason=%s",
            status,
            reason,
        )

    def _set_status(self, status: str, reason: str | None = None) -> None:
        if self.status == status and self.reason == reason:
            return
        self.status = status
        self.reason = reason
        self.status_callback(status, reason)

    @staticmethod
    def _stdout_heartbeat(payload: str) -> None:
        print(payload, flush=True)

    def _send_heartbeat_if_due(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_heartbeat_at < 1.0:
            return
        base = json.loads(
            HeartbeatMessage(
                worker="recorder",
                ts=self._last_ts,
                fps=self._frame_count / max(now - self._started_at, 1e-9),
                proc_ms=self._last_proc_ms,
            ).to_json()
        )
        # Optional health details let the supervisor report disk-full degradation.
        base.update(
            status=self.status,
            reason=self.reason,
            dropped_frames=self._dropped_frames,
        )
        self.heartbeat_callback(json.dumps(base, separators=(",", ":"), allow_nan=False))
        self._last_heartbeat_at = now

    def _create_session_directory(self) -> Path:
        root = Path(self.config.recording["directory"])
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = root / stamp
        suffix = 1
        while target.exists():
            target = root / f"{stamp}-{suffix:02d}"
            suffix += 1
        target.mkdir()
        snapshot = asdict(self.config)
        (target / "config.snapshot.yaml").write_text(
            yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        self.output_dir = target
        self._events = (target / "events.jsonl").open("w", encoding="utf-8", buffering=1)
        return target

    def _new_encoder(self) -> None:
        if self.output_dir is None:
            raise RuntimeError("Session directory is not initialized")
        self._encoder = VideoEncoder(
            str(self.config.recording.get("ffmpeg", "ffmpeg")),
            self.config.source.width,
            self.config.source.height,
            self.config.source.fps,
            self.output_dir / "video_%03d.mkv",
            segment_seconds=float(self.config.recording["segment_seconds"]),
            start_number=self._segment_start_number,
        )

    def _check_disk_space(self) -> bool:
        root = Path(self.config.recording["directory"])
        root.mkdir(parents=True, exist_ok=True)
        minimum_bytes = int(float(self.config.recording["min_free_gb"]) * 1024**3)
        free_bytes = shutil.disk_usage(root).free
        self._last_disk_check = time.monotonic()
        if free_bytes < minimum_bytes:
            self._set_status(
                "degraded", "free disk space is below the configured minimum; waiting for space"
            )
            self._send_heartbeat_if_due(force=True)
            return False
        if self.status == "degraded":
            self._set_status("ok", None)
        if self.output_dir is None:
            self._create_session_directory()
        return True

    def _start_encoder_if_needed(self) -> None:
        if self._encoder is None:
            self._new_encoder()

    def _write_jsonl(self, row: dict[str, Any]) -> None:
        if self._events is None:
            raise RuntimeError("JSONL file is not open")
        try:
            encoded = json.dumps(row, separators=(",", ":"), allow_nan=False)
        except ValueError:
            LOGGER.exception("Non-finite value in recorder row; replacing non-finite numbers")
            encoded = json.dumps(_json_safe(row), separators=(",", ":"), allow_nan=False)
        self._events.write(encoded + "\n")

    def _record_frame(self, frame: Frame, state_updates: list[dict[str, Any]]) -> None:
        started_ns = time.perf_counter_ns()
        if self.output_dir is None:
            self._create_session_directory()
        self._start_encoder_if_needed()
        assert self._encoder is not None
        gap = 0 if self._last_frame_id is None else max(0, frame.frame_id - self._last_frame_id - 1)
        if gap:
            self._dropped_frames += gap
            LOGGER.warning(
                "Frame bus skipped %s frame(s) before frame_id=%s",
                gap,
                frame.frame_id,
            )
            if self._last_frame_image is not None and self._last_frame_id is not None:
                for missing_id in range(self._last_frame_id + 1, frame.frame_id):
                    self._encoder.write(self._last_frame_image)
                    self._write_jsonl(
                        {
                            "frame_id": self._last_frame_id,
                            "ts": self._last_frame_ts,
                            "state_updates": [],
                            "events": [],
                            "repeat_fill": True,
                            "missing_frame_id": missing_id,
                            "dropped_before": 0,
                        }
                    )
                    self._frame_count += 1
        messages = [_json_safe(message) for message in state_updates]
        events = [event for message in messages for event in message.get("events", [])]
        states = []
        for message in messages:
            state = dict(message)
            state.pop("events", None)
            states.append(state)
        self._encoder.write(frame.image)
        self._write_jsonl(
            {
                "frame_id": frame.frame_id,
                "ts": frame.ts,
                "state_updates": states,
                "events": events,
                "repeat_fill": False,
                "missing_frame_ids": [],
                "dropped_before": gap,
            }
        )
        self._frame_count += 1
        self._last_frame_id = frame.frame_id
        self._last_frame_ts = frame.ts
        self._last_frame_image = frame.image
        self._last_ts = frame.ts
        self._last_proc_ms = (time.perf_counter_ns() - started_ns) / 1_000_000

    async def _read_game_states(self, ws_url: str) -> None:
        delay = 0.5
        while not self._stopping.is_set():
            try:
                async with websockets.connect(ws_url, max_size=4_000_000) as socket:
                    delay = 0.5
                    async for payload in socket:
                        try:
                            message = json.loads(payload, parse_constant=lambda _value: None)
                            if not isinstance(message, dict) or message.get("v") != 1:
                                raise ValueError("WebSocket message is not a Contract v1 object")
                            await self._state_messages.put(message)
                        except (json.JSONDecodeError, ValueError, TypeError) as exc:
                            LOGGER.warning("Ignoring invalid WebSocket GameState: %s", exc)
            except asyncio.CancelledError:
                raise
            except (subprocess.TimeoutExpired, TimeoutError) as exc:
                LOGGER.exception("Timed out while reading WebSocket state: %s", exc)
            except Exception:
                LOGGER.exception("WebSocket state reader failed; reconnecting")
            if not self._stopping.is_set():
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)

    def _take_state_updates(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        while True:
            try:
                messages.append(self._state_messages.get_nowait())
            except asyncio.QueueEmpty:
                return messages

    async def _heartbeat_loop(self) -> None:
        while not self._stopping.is_set():
            self._send_heartbeat_if_due(force=True)
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=1.0)
            except TimeoutError:
                pass

    def _close_encoder(self) -> None:
        if self._encoder is None:
            return
        encoder = self._encoder
        self._encoder = None
        try:
            encoder.close()
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            LOGGER.exception("Failed to finalize FFmpeg segment")
        self._segment_start_number = max(self._segment_start_number, self._frame_count)

    async def _wait_for_disk_space(self) -> None:
        retry_s = float(self.config.recording["disk_retry_interval_s"])
        while not self._stopping.is_set():
            self._send_heartbeat_if_due()
            try:
                if await asyncio.to_thread(self._check_disk_space):
                    return
            except Exception:
                LOGGER.exception("Disk space check failed; recorder remains degraded")
                self._set_status("degraded", "disk space check failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=retry_s)
            except TimeoutError:
                pass

    async def run(self) -> None:
        ffmpeg_setting = str(self.config.recording.get("ffmpeg", "ffmpeg"))
        ffmpeg = shutil.which(ffmpeg_setting)
        if ffmpeg is None and Path(ffmpeg_setting).is_file():
            ffmpeg = str(Path(ffmpeg_setting).resolve())
        if ffmpeg is None:
            self._set_status("degraded", "ffmpeg is not installed")
            LOGGER.error("FFmpeg is not installed")
            return
        bus: SharedFrameRing | None = None
        try:
            bus = SharedFrameRing(
                self.config.bus.name,
                self.config.source.width,
                self.config.source.height,
                self.config.bus.slots,
                create=False,
            )
            ws_url = f"ws://127.0.0.1:{self.config.server['ws_port']}/"
            state_task = asyncio.create_task(self._read_game_states(ws_url))
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            idle_wait = 1 / max(self.config.source.fps, 1.0)
            while not self._stopping.is_set():
                self._send_heartbeat_if_due()
                frame = await asyncio.to_thread(bus.read_latest)
                if frame is None or (
                    self._last_frame_id is not None and frame.frame_id <= self._last_frame_id
                ):
                    try:
                        await asyncio.wait_for(self._stopping.wait(), timeout=idle_wait)
                    except TimeoutError:
                        pass
                    continue
                try:
                    if time.monotonic() - self._last_disk_check >= float(
                        self.config.recording["disk_check_interval_s"]
                    ):
                        if not await asyncio.to_thread(self._check_disk_space):
                            self._close_encoder()
                            await self._wait_for_disk_space()
                            if self._stopping.is_set():
                                break
                    await asyncio.to_thread(self._record_frame, frame, self._take_state_updates())
                except Exception as exc:
                    LOGGER.exception("Recorder frame task failed at frame_id=%s", frame.frame_id)
                    self._set_status("degraded", f"recording error: {exc}")
                    self._close_encoder()
                    if isinstance(exc, OSError):
                        await self._wait_for_disk_space()
                    else:
                        try:
                            await asyncio.wait_for(self._stopping.wait(), timeout=idle_wait)
                        except TimeoutError:
                            pass
                    self._send_heartbeat_if_due(force=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("Recorder run failed; entering degraded recovery")
            self._set_status("degraded", f"recorder setup error: {exc}")
        finally:
            self._stopping.set()
            task = locals().get("state_task")
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    LOGGER.exception("WebSocket state task failed during shutdown")
            heartbeat_task = locals().get("heartbeat_task")
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    LOGGER.exception("Recorder heartbeat task failed during shutdown")
            self._close_encoder()
            if self._events is not None:
                try:
                    self._events.close()
                except Exception:
                    LOGGER.exception("Failed to close events.jsonl")
                self._events = None
            if bus is not None:
                bus.close()
            self._send_heartbeat_if_due(force=True)

    def stop(self) -> None:
        self._stopping.set()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


async def serve(config_path: str | None = None) -> None:
    config = load_config(config_path)
    await Recorder(config, config_snapshot=config_path).run()


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Record the latest frame bus and GameState stream")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    asyncio.run(serve(args.config))


if __name__ == "__main__":
    main()
