"""Send simulated Contract v1 GameState messages to the overlay."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hud.config import load_config


def _player(
    pid: int, team: int, name: str, color: str, court: list[float], phase: float, elapsed_s: float
) -> dict[str, Any]:
    """Synthetic player wandering around a court anchor; img_xy is a rough fake projection."""
    x = court[0] + 0.8 * math.sin(phase + pid)
    y = court[1] + 0.5 * math.cos(phase + pid * 1.3)
    foot = [1280 * (0.15 + 0.7 * y / 6.1), 720 * (0.85 - 0.45 * x / 13.4)]
    return {
        "id": pid,
        "name": name,
        "color": color,
        "team": team,
        "side": team,
        "tracked": True,
        "img_xy": foot,
        "bbox": [foot[0] - 40, foot[1] - 220, foot[0] + 40, foot[1]],
        "court_xy_m": [x, y],
        "speed_mps": 2.2 + 0.3 * math.sin(phase * 2 + pid),
        "dist_m": 18.0 + pid + elapsed_s * 0.4,
    }


def _untrack(player: dict[str, Any]) -> None:
    """Contract: untracked players stay in the list with null positions."""
    player.update(tracked=False, img_xy=None, bbox=None, court_xy_m=None, speed_mps=None)


def make_state(ts: int, elapsed_s: float, mode: str = "singles") -> dict[str, Any]:
    """Build one clearly synthetic snapshot (singles or doubles) for local UI development."""
    phase = elapsed_s * 0.8
    if mode == "doubles":
        players = [
            _player(0, 0, "TEAM A ONE", "#39e4f5", [2.2, 1.6], phase, elapsed_s),
            _player(1, 0, "TEAM A TWO", "#5ff0a9", [4.6, 4.5], phase, elapsed_s),
            _player(2, 1, "TEAM B ONE", "#b693ff", [9.0, 1.6], phase, elapsed_s),
            _player(3, 1, "TEAM B TWO", "#ff8fc7", [11.2, 4.5], phase, elapsed_s),
        ]
        # Drop player 3 for 3 s out of every 10 s to exercise the untracked path.
        if elapsed_s % 10 > 7:
            _untrack(players[3])
    else:
        players = [
            _player(0, 0, "PLAYER ONE", "#39e4f5", [3.2, 2.5], phase, elapsed_s),
            _player(1, 1, "PLAYER TWO", "#b693ff", [10.1, 3.5], phase, elapsed_s),
        ]
    shuttle_px = [640 + 250 * math.sin(phase * 1.7), 260 + 120 * math.cos(phase * 1.7)]
    rally_s = elapsed_s % 20
    # Landing estimate exists only at the end of a rally; null while the shuttle flies.
    landing = [11.6, 1.2] if rally_s > 16 else None
    return {
        "v": 1,
        "ts": ts,
        "match": {"mode": mode, "score": [11, 9], "games": [1, 0], "server": 0},
        "rally": {"active": landing is None, "n": 21, "t_ms": int(rally_s * 1000), "shots": 5},
        "players": players,
        "shuttle": {
            "status": "tracked",
            "img_xy": shuttle_px,
            "landing_xy_m_est": landing,
            "last_shot": {"by": 1, "speed_kmh_est": 150, "type": "clear"},
        },
        "events": [],
        "health": {
            "fps": 60.0,
            "latency_ms_p50": 128.0,
            "latency_ms_p99": 190.0,
            "dropped": 0,
            "workers": {"players": "ok", "shuttle": "ok", "audio": "ok"},
        },
    }


async def serve(host: str, port: int, mode: str) -> None:
    started = time.monotonic()

    async def handler(connection: Any) -> None:
        try:
            while True:
                state = make_state(time.monotonic_ns(), time.monotonic() - started, mode)
                await connection.send(json.dumps(state, separators=(",", ":")))
                await asyncio.sleep(1 / 30)
        except websockets.exceptions.ConnectionClosed:
            return

    async with websockets.serve(handler, host, port, max_size=1_000_000):
        print(f"Mock GameState WebSocket listening on ws://{host}:{port}", flush=True)
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML application config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--mode", choices=["singles", "doubles"], default="singles")
    args = parser.parse_args()
    config = load_config(args.config)
    asyncio.run(serve(args.host, args.port or config.server["ws_port"], args.mode))


if __name__ == "__main__":
    main()
