"""Independent frame-bus recorder with segmented FFmpeg video and JSONL state."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
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

StatusCallback = Callable[[str, str | None], None]


def empty_game_state(ts: int) -> dict[str, Any]:
    """Return a Contract v1 state with explicit nulls before fusion connects."""
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
            "court_xy_m": None,
            "last_shot": {"by": None, "speed_kmh_est": None, "type": None},
        },
        "events": [],
        "health": {
            "fps": 0.0,
            "latency_ms_p50": None,
            "latency_ms_p99": None,
            "dropped": 0,
            "workers": {"players": "down", "shuttle": "down", "audio": "down"},
        },
    }


class VideoEncoder:
    """Pipe raw BGR frames to NVENC when a small probe succeeds, else x264."""

    def __init__(self, ffmpeg: str, width: int, height: int, fps: float, output: Path) -> None:
        self.ffmpeg = ffmpeg
        self.width = width
        self.height = height
        self.fps = fps
        self.output = output
        self.codec = "h264_nvenc" if self._probe_nvenc() else "libx264"
        self.process: subprocess.Popen[bytes] | None = None
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
            return False
        return result.returncode == 0

    def start(self) -> None:
        command = [
            self.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            self.codec,
        ]
        if self.codec == "h264_nvenc":
            command.extend(["-preset", "p4"])
        else:
            command.extend(["-preset", "veryfast", "-crf", "23"])
        command.extend(["-movflags", "+faststart", str(self.output)])
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg did not open a raw-frame input pipe")

    def write(self, frame: bytes) -> None:
        if self.process is None or self.process.stdin is None or self.process.poll() is not None:
            raise RuntimeError("Video encoder stopped unexpectedly")
        self.process.stdin.write(frame)

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        return_code = self.process.wait(timeout=30)
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
        self.config_snapshot = Path(config_snapshot) if config_snapshot else None
        self.status_callback = status_callback or self._log_status
        self.heartbeat_callback = heartbeat_callback or self._stdout_heartbeat
        self.status = "ok"
        self.reason: str | None = None
        self.latest_state: dict[str, Any] | None = None
        self._stopping = asyncio.Event()
        self._last_frame_id = 0
        self._frame_count = 0
        self._segment_index = 0
        self._segment_start_ts: int | None = None
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
        message = {"component": "recorder", "status": status, "reason": reason}
        print(json.dumps(message), file=sys.stderr, flush=True)

    def _set_status(self, status: str, reason: str | None = None) -> None:
        if self.status == status and self.reason == reason:
            return
        self.status = status
        self.reason = reason
        self.status_callback(status, reason)

    @staticmethod
    def _stdout_heartbeat(payload: str) -> None:
        print(payload, flush=True)

    def _send_heartbeat_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat_at < 1.0:
            return
        self.heartbeat_callback(
            HeartbeatMessage(
                worker="recorder",
                ts=self._last_ts,
                fps=self._frame_count / max(now - self._started_at, 1e-9),
                proc_ms=self._last_proc_ms,
            ).to_json()
        )
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
        if self.config_snapshot is not None and self.config_snapshot.exists():
            shutil.copyfile(self.config_snapshot, target / "config.snapshot.yaml")
        else:
            snapshot = asdict(self.config)
            (target / "config.snapshot.yaml").write_text(
                yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
        self.output_dir = target
        self._events = (target / "events.jsonl").open("w", encoding="utf-8", buffering=1)
        return target

    def _new_encoder(self) -> None:
        assert self.output_dir is not None
        suffix = "" if self._segment_index == 0 else f"_{self._segment_index:03d}"
        output = self.output_dir / f"video{suffix}.mp4"
        self._encoder = VideoEncoder(
            shutil.which("ffmpeg") or "ffmpeg",
            self.config.source.width,
            self.config.source.height,
            self.config.source.fps,
            output,
        )
        self._segment_index += 1

    def _check_disk_space(self) -> bool:
        assert self.output_dir is not None
        minimum_bytes = int(float(self.config.recording["min_free_gb"]) * 1024**3)
        free_bytes = shutil.disk_usage(self.output_dir).free
        if free_bytes < minimum_bytes:
            self._set_status(
                "degraded", "recording stopped: free disk space is below configured minimum"
            )
            return False
        self._last_disk_check = time.monotonic()
        return True

    def _record_frame(self, frame: Frame, state: dict[str, Any]) -> None:
        started_ns = time.perf_counter_ns()
        if self.output_dir is None:
            self._create_session_directory()
        if time.monotonic() - self._last_disk_check >= 5.0 and not self._check_disk_space():
            raise OSError(
                "Recorder stopped because disk free space is below its configured minimum"
            )
        if self._encoder is None:
            self._new_encoder()
            self._segment_start_ts = frame.ts
        elif self._segment_start_ts is not None:
            segment_seconds = float(self.config.recording.get("segment_seconds", 1800))
            if frame.ts - self._segment_start_ts >= int(segment_seconds * 1_000_000_000):
                self._encoder.close()
                self._encoder = None
                self._new_encoder()
                self._segment_start_ts = frame.ts
        assert self._encoder is not None and self._events is not None
        self._encoder.write(frame.image)
        self._events.write(json.dumps(state, separators=(",", ":"), allow_nan=False) + "\n")
        self._frame_count += 1
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
                            message = json.loads(payload)
                            if message.get("v") == 1:
                                self.latest_state = message
                        except (json.JSONDecodeError, AttributeError):
                            continue
            except (OSError, websockets.exceptions.WebSocketException):
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)

    async def run(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self._set_status("degraded", "ffmpeg is not installed")
            raise RuntimeError("FFmpeg is required to record MP4 video")
        bus = SharedFrameRing(
            self.config.bus.name,
            self.config.source.width,
            self.config.source.height,
            self.config.bus.slots,
            create=False,
        )
        ws_url = f"ws://127.0.0.1:{self.config.server['ws_port']}/"
        state_task = asyncio.create_task(self._read_game_states(ws_url))
        idle_wait = 1 / max(self.config.source.fps, 1.0)
        try:
            while not self._stopping.is_set() and self.status == "ok":
                self._send_heartbeat_if_due()
                frame = await asyncio.to_thread(bus.read_latest)
                if frame is None or frame.frame_id <= self._last_frame_id:
                    await asyncio.sleep(idle_wait)
                    continue
                self._last_frame_id = frame.frame_id
                game_state = self.latest_state or empty_game_state(frame.ts)
                if time.monotonic() - self._last_disk_check >= 5.0 and self.output_dir is not None:
                    if not await asyncio.to_thread(self._check_disk_space):
                        break
                try:
                    await asyncio.to_thread(self._record_frame, frame, game_state)
                except (OSError, RuntimeError) as exc:
                    self._set_status("degraded", str(exc))
                    break
        finally:
            self._stopping.set()
            state_task.cancel()
            if self._encoder is not None:
                try:
                    await asyncio.to_thread(self._encoder.close)
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    self._set_status("degraded", f"video encoder failed to finalize: {exc}")
                self._encoder = None
            if self._events is not None:
                self._events.close()
                self._events = None
            bus.close()

    def stop(self) -> None:
        self._stopping.set()


async def serve(config_path: str | None = None) -> None:
    config = load_config(config_path)
    await Recorder(config, config_snapshot=config_path).run()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Record the latest frame bus and GameState stream")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    asyncio.run(serve(args.config))


if __name__ == "__main__":
    main()
