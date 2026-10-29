"""Contract v1 GameState assembly and operator state updates."""

from __future__ import annotations

import json
import statistics
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from hud.bus.messages import HeartbeatMessage, HitMessage, PlayersMessage, ShuttleMessage
from hud.fusion.homography import load_homography, transform_point

_WORKERS = ("players", "shuttle", "audio")


def validate_game_state(value: dict[str, Any]) -> None:
    """Raise ValueError when a GameState is missing or violates Contract v1 fields."""
    if value.get("v") != 1 or not isinstance(value.get("ts"), int):
        raise ValueError("GameState requires integer v=1 and monotonic ts")
    if set(value) != {"v", "ts", "match", "rally", "players", "shuttle", "events", "health"}:
        raise ValueError("GameState top-level fields do not match Contract v1")
    match = value["match"]
    if (
        match.get("mode") != "singles"
        or len(match.get("score", [])) != 2
        or len(match.get("games", [])) != 2
    ):
        raise ValueError("Invalid match fields")
    if not {"active", "n", "t_ms", "shots"}.issubset(value["rally"]):
        raise ValueError("Invalid rally fields")
    if len(value["players"]) != 2:
        raise ValueError("Singles GameState must contain exactly two player slots")
    for player_id, player in enumerate(value["players"]):
        required = {
            "id",
            "name",
            "color",
            "side",
            "tracked",
            "img_xy",
            "bbox",
            "court_xy_m",
            "speed_mps",
            "dist_m",
        }
        if set(player) != required or player["id"] != player_id:
            raise ValueError("Invalid player schema")
    shuttle = value["shuttle"]
    if set(shuttle) != {"status", "img_xy", "court_xy_m", "last_shot"}:
        raise ValueError("Invalid shuttle fields")
    if shuttle["status"] not in {"tracked", "predicted", "lost"}:
        raise ValueError("Invalid shuttle status")
    if not {"fps", "latency_ms_p50", "latency_ms_p99", "dropped", "workers"}.issubset(
        value["health"]
    ):
        raise ValueError("Invalid health fields")
    if not isinstance(value["events"], list):
        raise ValueError("events must be an array")
    json.dumps(value)


class FusionState:
    """Thread-safe mutable match snapshot; unknown Contract fields remain explicit nulls."""

    def __init__(self, calibration_path: str | Path = "config/court_calib.json") -> None:
        self._lock = threading.RLock()
        self.calibration_path = Path(calibration_path)
        self.homography = load_homography(self.calibration_path)
        self._workers = {
            name: {"status": "down", "last_seen": 0.0, "fps": None} for name in _WORKERS
        }
        self._latencies_ms: deque[float] = deque(maxlen=256)
        self._events: list[dict[str, Any]] = []
        self._track_to_id: dict[int, int] = {}
        self._player_config: dict[int, dict[str, Any]] = {}
        self._players: dict[int, dict[str, Any]] = {}
        self._player_history: dict[int, tuple[int, float, float]] = {}
        self._score: list[int | None] = [None, None]
        self._games: list[int | None] = [None, None]
        self._server: int | None = None
        self._shuttle: dict[str, Any] = {
            "status": "lost",
            "img_xy": None,
            "court_xy_m": None,
            "last_shot": {"by": None, "speed_kmh_est": None, "type": None},
        }
        self._rally = {"active": False, "n": 0, "t_ms": 0, "shots": 0}
        self._ts = 0
        self._last_frame_id: int | None = None
        self._dropped = 0
        self._fps = 0.0
        self._frame_times: deque[float] = deque(maxlen=120)

    def _record_latency(self, ts: int) -> None:
        now_ns = time.monotonic_ns()
        if 0 < ts <= now_ns:
            self._latencies_ms.append((now_ns - ts) / 1_000_000)
        self._ts = max(self._ts, ts)

    def ingest(self, payload: str | bytes) -> None:
        message = json.loads(payload)
        kind = message.get("type")
        now = time.monotonic()
        with self._lock:
            if kind == "players":
                decoded = PlayersMessage.from_json(payload)
                self._record_latency(decoded.ts)
                if self._last_frame_id is not None and decoded.frame_id > self._last_frame_id + 1:
                    self._dropped += decoded.frame_id - self._last_frame_id - 1
                self._last_frame_id = decoded.frame_id
                self._frame_times.append(now)
                if len(self._frame_times) > 1:
                    elapsed = self._frame_times[-1] - self._frame_times[0]
                    self._fps = (len(self._frame_times) - 1) / elapsed if elapsed else 0.0
                for player in self._players.values():
                    player["tracked"] = False
                for detection in decoded.players:
                    track_id = detection.track_id
                    player_id = self._track_to_id.setdefault(track_id, len(self._track_to_id))
                    if player_id > 1:
                        continue
                    court_xy = (
                        transform_point(self.homography, detection.foot_px)
                        if self.homography is not None
                        else None
                    )
                    speed: float | None = None
                    distance: float | None = None
                    previous = self._player_history.get(player_id)
                    if previous and court_xy:
                        prev_ts, prev_x, prev_y = previous
                        elapsed_s = (decoded.ts - prev_ts) / 1_000_000_000
                        if elapsed_s > 0:
                            measured = (
                                (court_xy[0] - prev_x) ** 2 + (court_xy[1] - prev_y) ** 2
                            ) ** 0.5 / elapsed_s
                            old = self._players.get(player_id, {}).get("speed_mps")
                            speed = measured if old is None else 0.35 * measured + 0.65 * old
                            old_dist = self._players.get(player_id, {}).get("dist_m")
                            step = (
                                (court_xy[0] - prev_x) ** 2 + (court_xy[1] - prev_y) ** 2
                            ) ** 0.5
                            distance = (old_dist or 0.0) + step
                    self._player_history[player_id] = (decoded.ts, *(court_xy or (0.0, 0.0)))
                    metadata = self._player_config.get(player_id, {})
                    self._players[player_id] = {
                        "id": player_id,
                        "name": metadata.get("name"),
                        "color": metadata.get("color"),
                        "side": metadata.get("side"),
                        "tracked": True,
                        "img_xy": detection.foot_px,
                        "bbox": detection.bbox,
                        "court_xy_m": court_xy,
                        "speed_mps": speed,
                        "dist_m": distance,
                    }
            elif kind == "shuttle":
                decoded = ShuttleMessage.from_json(payload)
                self._record_latency(decoded.ts)
                court_xy = (
                    transform_point(self.homography, decoded.px)
                    if decoded.visible and decoded.px and self.homography is not None
                    else None
                )
                self._shuttle.update(
                    status="tracked" if decoded.visible else "lost",
                    img_xy=decoded.px if decoded.visible else None,
                    court_xy_m=court_xy,
                )
            elif kind == "hit":
                decoded = HitMessage.from_json(payload)
                self._record_latency(decoded.ts)
                self._events.append(
                    {"ts": decoded.ts, "kind": "shot", "data": {"strength": decoded.strength}}
                )
            elif kind == "heartbeat":
                decoded = HeartbeatMessage.from_json(payload)
                if decoded.worker not in _WORKERS:
                    return
                self._workers.setdefault(
                    decoded.worker, {"status": "down", "last_seen": 0.0}
                ).update(status="ok", last_seen=now, fps=decoded.fps)
                self._record_latency(decoded.ts)

    def calibrate(self, corners_px: list[list[float]]) -> None:
        from hud.fusion.homography import save_homography

        with self._lock:
            self.homography = save_homography(self.calibration_path, corners_px)

    def set_players(self, players: list[dict[str, Any]]) -> None:
        with self._lock:
            self._player_config = {int(player["id"]): dict(player) for player in players}
            for player_id, value in self._players.items():
                metadata = self._player_config.get(player_id, {})
                value.update(
                    name=metadata.get("name"),
                    color=metadata.get("color"),
                    side=metadata.get("side"),
                )

    def add_score(self, side: int, delta: int) -> None:
        if side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        with self._lock:
            self._score[side] = (self._score[side] or 0) + delta

    def swap_ids(self, first: int, second: int) -> None:
        if first not in (0, 1) or second not in (0, 1):
            raise ValueError("player IDs must be 0 or 1")
        with self._lock:
            self._track_to_id = {
                track_id: second
                if player_id == first
                else first
                if player_id == second
                else player_id
                for track_id, player_id in self._track_to_id.items()
            }
            self._player_history[first], self._player_history[second] = (
                self._player_history.get(second),
                self._player_history.get(first),
            )
            self._players[first], self._players[second] = (
                self._players.get(second),
                self._players.get(first),
            )
            if self._players[first] is None:
                self._players.pop(first)
            else:
                self._players[first]["id"] = first
            if self._players[second] is None:
                self._players.pop(second)
            else:
                self._players[second]["id"] = second
            first_meta = self._player_config.get(first)
            second_meta = self._player_config.get(second)
            if first_meta is not None or second_meta is not None:
                self._player_config[first], self._player_config[second] = (
                    second_meta or {},
                    first_meta or {},
                )

    def snapshot(self, *, drain_events: bool = True) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            health_workers: dict[str, str] = {}
            for name, worker in self._workers.items():
                if now - worker["last_seen"] > 3.0:
                    worker["status"] = "down"
                health_workers[name] = worker["status"]
            latencies = sorted(self._latencies_ms)
            p50 = statistics.median(latencies) if latencies else None
            p99 = (
                latencies[min(int((len(latencies) - 1) * 0.99), len(latencies) - 1)]
                if latencies
                else None
            )
            events = self._events
            if drain_events:
                self._events = []
            players = [self._players.get(index, self._empty_player(index)) for index in range(2)]
            result = {
                "v": 1,
                "ts": self._ts,
                "match": {
                    "mode": "singles",
                    "score": list(self._score),
                    "games": list(self._games),
                    "server": self._server,
                },
                "rally": dict(self._rally),
                "players": players,
                "shuttle": json.loads(json.dumps(self._shuttle)),
                "events": list(events),
                "health": {
                    "fps": self._fps,
                    "latency_ms_p50": p50,
                    "latency_ms_p99": p99,
                    "dropped": self._dropped,
                    "workers": health_workers,
                },
            }
            validate_game_state(result)
            return result

    @staticmethod
    def _empty_player(player_id: int) -> dict[str, Any]:
        return {
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
