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


def make_state(ts: int, elapsed_s: float) -> dict[str, Any]:
    """Build one clearly synthetic singles snapshot for local UI development."""
    phase = elapsed_s * 0.8
    players = [
        {
            "id": 0,
            "name": "PLAYER ONE",
            "color": "#39e4f5",
            "side": 0,
            "tracked": True,
            "img_xy": [330 + 110 * math.sin(phase), 590 + 30 * math.cos(phase)],
            "bbox": [280, 380, 380, 620],
            "court_xy_m": [3.2 + 0.8 * math.sin(phase), 2.0 + 0.4 * math.cos(phase)],
            "speed_mps": 2.4 + 0.3 * math.sin(phase * 2),
            "dist_m": 18.6 + elapsed_s * 0.4,
        },
        {
            "id": 1,
            "name": "PLAYER TWO",
            "color": "#b693ff",
            "side": 1,
            "tracked": True,
            "img_xy": [950 + 90 * math.sin(phase + 2), 570 + 36 * math.cos(phase + 1)],
            "bbox": [900, 350, 1000, 610],
            "court_xy_m": [10.1 + 0.7 * math.sin(phase + 2), 4.0 + 0.5 * math.cos(phase + 1)],
            "speed_mps": 2.1 + 0.2 * math.sin(phase * 2 + 1),
            "dist_m": 21.4 + elapsed_s * 0.35,
        },
    ]
    shuttle_px = [640 + 250 * math.sin(phase * 1.7), 260 + 120 * math.cos(phase * 1.7)]
    return {
        "v": 1,
        "ts": ts,
        "match": {"mode": "singles", "score": [11, 9], "games": [1, 0], "server": 0},
        "rally": {"active": True, "n": 21, "t_ms": int((elapsed_s % 20) * 1000), "shots": 5},
        "players": players,
        "shuttle": {
            "status": "tracked",
            "img_xy": shuttle_px,
            "court_xy_m": [6.7 + 4.5 * math.sin(phase), 3.05 + 1.4 * math.cos(phase)],
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


async def serve(host: str, port: int) -> None:
    started = time.monotonic()

    async def handler(connection: Any) -> None:
        try:
            while True:
                state = make_state(time.monotonic_ns(), time.monotonic() - started)
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
    args = parser.parse_args()
    config = load_config(args.config)
    asyncio.run(serve(args.host, args.port or config.server["ws_port"]))


if __name__ == "__main__":
    main()
