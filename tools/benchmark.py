"""Replay a clip through the full system and check G1 (docs/GOALS.md).

Latency is `time.monotonic_ns()` at WS receive minus GameState `ts` (monotonic ns at
capture), so it is only meaningful when this tool runs on the same host as the system.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# G1 thresholds from docs/GOALS.md.
G1_MIN_FPS = 60.0
G1_MAX_LATENCY_P50_MS = 150.0
G1_MAX_LATENCY_P99_MS = 250.0

STARTUP_TIMEOUT_S = 90.0  # model load can take up to supervisor.startup_grace_s per worker
CAPTURE_COMMAND = [sys.executable, "-m", "hud.sources.capture"]


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100); None for no samples."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def g1_failures(fps: float | None, p50_ms: float | None, p99_ms: float | None) -> list[str]:
    """Return the G1 checks that failed; a missing measurement is a failure."""
    failures = []
    if fps is None or fps < G1_MIN_FPS:
        failures.append(f"fps {fps} < {G1_MIN_FPS}")
    if p50_ms is None or p50_ms > G1_MAX_LATENCY_P50_MS:
        failures.append(f"latency p50 {p50_ms} ms > {G1_MAX_LATENCY_P50_MS} ms")
    if p99_ms is None or p99_ms > G1_MAX_LATENCY_P99_MS:
        failures.append(f"latency p99 {p99_ms} ms > {G1_MAX_LATENCY_P99_MS} ms")
    return failures


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config(clip: Path, config_path: Path | None, *, loop: bool) -> dict[str, Any]:
    """Default config + user config + replay source + a supervised capture worker."""
    with (ROOT / "config" / "default.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if config_path is not None:
        with config_path.open(encoding="utf-8") as stream:
            config = _merge(config, yaml.safe_load(stream) or {})
    config = _merge(config, {"source": {"type": "file", "path": str(clip), "loop": loop}})
    # The supervisor only launches `workers`; capture heartbeats on stdout like a worker.
    config["workers"].setdefault("capture", {"enabled": True, "command": CAPTURE_COMMAND})
    return config


class System:
    """Production launch path: `hud.supervisor` (workers + capture) and the HTTP/WS server.

    Children load config via HUD_DEFAULT_CONFIG, so every process sees the same file.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        handle, name = tempfile.mkstemp(prefix="hud-bench-", suffix=".yaml")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream)
        self.config_file = Path(name)
        self.env = dict(os.environ, HUD_DEFAULT_CONFIG=name, PYTHONUNBUFFERED="1")
        self.env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")])
        )
        self.processes: list[subprocess.Popen[bytes]] = []

    def start(self) -> None:
        for module in ("hud.supervisor", "hud.server.operator_api"):
            self.processes.append(
                subprocess.Popen([sys.executable, "-m", module], cwd=ROOT, env=self.env)
            )

    @property
    def supervisor_pid(self) -> int:
        return self.processes[0].pid

    def pids(self) -> list[int]:
        """Every live process of the system (roots and descendants); needs psutil."""
        import psutil

        pids = []
        for process in self.processes:
            try:
                root = psutil.Process(process.pid)
                pids += [root.pid] + [child.pid for child in root.children(recursive=True)]
            except psutil.NoSuchProcess:
                continue
        return pids

    def alive(self) -> bool:
        return all(process.poll() is None for process in self.processes)

    def stop(self) -> None:
        # Terminating the supervisor alone would orphan its workers, so kill whole trees.
        try:
            import psutil
        except ImportError:
            psutil = None
        for process in self.processes:
            if psutil is not None:
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(children, timeout=5)
            process.kill()
            process.wait()
        try:
            self.config_file.unlink(missing_ok=True)
        except OSError:  # Windows: a dying child may still hold it; it is only a temp file
            pass


def nvidia_smi(query: str) -> list[list[str]] | None:
    """Rows of `nvidia-smi --query-*` CSV output, or None without a usable nvidia-smi."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        output = subprocess.run(
            [executable, query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [[cell.strip() for cell in line.split(",")] for line in output.splitlines() if line]


def gpu_sample() -> tuple[float, float] | None:
    """(utilisation %, VRAM used MiB) of GPU 0, whole device."""
    rows = nvidia_smi("--query-gpu=utilization.gpu,memory.used")
    try:
        return (float(rows[0][0]), float(rows[0][1])) if rows else None
    except (IndexError, ValueError):
        return None


def get_health(port: int) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


async def _heartbeats(workers: list[str], fps: dict[str, float]) -> None:
    """Last heartbeat fps per worker from the same PUB sockets fusion subscribes to."""
    import zmq
    import zmq.asyncio

    context = zmq.asyncio.Context.instance()
    poller = zmq.asyncio.Poller()
    for worker in workers:
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.connect(f"ipc:///tmp/hud_{worker}")
        poller.register(socket, zmq.POLLIN)
    while True:
        for socket, _ in await poller.poll(timeout=100):
            try:
                message = json.loads(await socket.recv_string())
            except ValueError:
                continue
            if message.get("type") == "heartbeat" and message.get("fps") is not None:
                fps[message["worker"]] = float(message["fps"])


async def collect(
    config: dict[str, Any], seconds: float, alive: Callable[[], bool]
) -> dict[str, Any]:
    import websockets

    ws_port = config["server"]["ws_port"]
    http_port = config["server"]["http_port"]
    zmq_workers = [
        name
        for name, worker in config["workers"].items()
        if worker.get("enabled") and name != "capture"
    ]
    worker_fps: dict[str, float] = {}
    try:
        import zmq

        ipc = zmq.has("ipc")
    except ImportError:
        ipc = False
    # Without ipc (e.g. Windows libzmq builds) per-worker fps stays null.
    heartbeat_task = asyncio.create_task(_heartbeats(zmq_workers, worker_fps)) if ipc else None

    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while True:
        try:
            websocket = await websockets.connect(f"ws://127.0.0.1:{ws_port}/")
            break
        except OSError:
            if not alive():
                raise RuntimeError("a system process exited during startup") from None
            if time.monotonic() > deadline:
                raise TimeoutError(f"WS :{ws_port} did not come up") from None
            await asyncio.sleep(0.5)

    latencies_ms: list[float] = []
    health_samples: list[dict[str, Any]] = []
    gpu_samples: list[tuple[float, float]] = []
    last_ts = 0
    end = None
    next_poll = 0.0
    async with websocket:
        while end is None or time.monotonic() < end:
            if time.monotonic() > deadline and end is None:
                raise TimeoutError("no captured frame reached the WS stream")
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=1)
            except TimeoutError:
                raw = None
            received_ns = time.monotonic_ns()
            if raw is not None:
                ts = json.loads(raw)["ts"]
                # WS sends at 30 Hz; only the first snapshot carrying a new frame counts.
                if ts > last_ts:
                    if end is None:
                        end = time.monotonic() + seconds  # measure from first real frame
                    else:
                        latencies_ms.append((received_ns - ts) / 1e6)
                    last_ts = ts
            if end is not None and time.monotonic() >= next_poll:
                next_poll = time.monotonic() + 1
                health = await asyncio.to_thread(get_health, http_port)
                if health is not None:
                    health_samples.append(health)
                gpu = await asyncio.to_thread(gpu_sample)
                if gpu is not None:
                    gpu_samples.append(gpu)
    if heartbeat_task is not None:
        heartbeat_task.cancel()

    fps_values = [h["fps"] for h in health_samples if h.get("fps")]
    dropped = [h["dropped"] for h in health_samples if h.get("dropped") is not None]
    return {
        "seconds": seconds,
        "fps": statistics.median(fps_values) if fps_values else None,
        "worker_fps": worker_fps or None,
        "latency_ms_p50": percentile(latencies_ms, 50),
        "latency_ms_p99": percentile(latencies_ms, 99),
        "latency_samples": len(latencies_ms),
        "dropped": dropped[-1] - dropped[0] if dropped else None,
        "workers_status": health_samples[-1]["workers"] if health_samples else None,
        "gpu_util_pct": statistics.mean(g[0] for g in gpu_samples) if gpu_samples else None,
        "vram_used_mib_max": max(g[1] for g in gpu_samples) if gpu_samples else None,
        "note": "latency = monotonic_ns at WS receive - GameState ts; same host only",
    }


def print_table(result: dict[str, Any]) -> None:
    rows = [
        ("fps (fusion)", result["fps"], f">= {G1_MIN_FPS}"),
        ("latency p50 ms", result["latency_ms_p50"], f"<= {G1_MAX_LATENCY_P50_MS}"),
        ("latency p99 ms", result["latency_ms_p99"], f"<= {G1_MAX_LATENCY_P99_MS}"),
        ("latency samples", result["latency_samples"], ""),
        ("dropped frames", result["dropped"], ""),
        ("GPU util %", result["gpu_util_pct"], ""),
        ("VRAM max MiB", result["vram_used_mib_max"], ""),
    ]
    for name, fps in (result["worker_fps"] or {}).items():
        rows.append((f"fps {name}", fps, ""))
    for name, value, target in rows:
        shown = f"{value:.1f}" if isinstance(value, float) else str(value)
        print(f"{name:<18}{shown:>12}  {target}")
    print("G1: " + ("PASS" if result["g1_pass"] else "FAIL " + "; ".join(result["g1_failures"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path, help="Clip to replay")
    parser.add_argument("--config", type=Path, help="YAML merged over config/default.yaml")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out", type=Path, help="Write the result JSON here")
    args = parser.parse_args()
    if not args.replay.is_file():
        parser.error(f"clip not found: {args.replay}")

    config = build_config(args.replay.resolve(), args.config, loop=True)
    system = System(config)
    system.start()
    try:
        result = asyncio.run(collect(config, args.seconds, system.alive))
    except Exception as exc:  # the CLI reports any abort the same way
        print(f"benchmark aborted: {exc!r}", file=sys.stderr)
        return 2
    finally:
        system.stop()
    failures = g1_failures(result["fps"], result["latency_ms_p50"], result["latency_ms_p99"])
    result.update(clip=str(args.replay), g1_pass=not failures, g1_failures=failures)
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_table(result)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
