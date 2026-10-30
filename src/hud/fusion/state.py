"""Contract v1 GameState assembly and operator state updates."""

from __future__ import annotations

import json
import math
import statistics
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from scipy.optimize import linear_sum_assignment

from hud.bus.messages import HeartbeatMessage, HitMessage, PlayersMessage, ShuttleMessage
from hud.fusion.homography import COURT_LENGTH_M, load_homography, transform_point
from hud.fusion.kalman import ShuttleKalmanFilter
from hud.fusion.rally_fsm import RallyFSM
from hud.fusion.stats import ShotDetector, ShotInfo

_WORKERS = ("players", "shuttle", "audio")
_NET_X_M = COURT_LENGTH_M / 2
_HEARTBEAT_TIMEOUT_S = 3.0


def _min_cost_pairs(costs: list[list[float]]) -> list[tuple[int, int]]:
    """Exact rectangular assignment (Hungarian); returns (row, col) pairs."""
    # The players worker forwards everyone in the ROI, so detections can outnumber slots.
    if not costs or not costs[0]:
        return []
    rows, cols = linear_sum_assignment(costs)
    return list(zip(rows.tolist(), cols.tolist(), strict=True))


def validate_game_state(value: dict[str, Any]) -> None:
    """Raise ValueError when a GameState is missing or violates Contract v1 fields."""
    if value.get("v") != 1 or not isinstance(value.get("ts"), int):
        raise ValueError("GameState requires integer v=1 and monotonic ts")
    if set(value) != {"v", "ts", "match", "rally", "players", "shuttle", "events", "health"}:
        raise ValueError("GameState top-level fields do not match Contract v1")
    match = value["match"]
    mode = match.get("mode")
    if (
        mode not in {"singles", "doubles"}
        or len(match.get("score", [])) != 2
        or len(match.get("games", [])) != 2
    ):
        raise ValueError("Invalid match fields")
    if not {"active", "n", "t_ms", "shots"}.issubset(value["rally"]):
        raise ValueError("Invalid rally fields")
    expected_players = 2 if mode == "singles" else 4
    if len(value["players"]) != expected_players:
        raise ValueError(f"{mode.title()} GameState must contain {expected_players} player slots")
    for player_id, player in enumerate(value["players"]):
        required = {
            "id",
            "name",
            "color",
            "team",
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
        if player["side"] not in (0, 1) or player["team"] not in (0, 1):
            raise ValueError("Every player side and team must be 0 or 1")
        court_xy = player["court_xy_m"]
        if court_xy is not None:
            lower_y, upper_y = (0.0, 6.10) if mode == "doubles" else (0.46, 5.64)
            if len(court_xy) != 2 or not lower_y <= court_xy[1] <= upper_y:
                raise ValueError(f"Player court y is outside the {mode} sidelines")
    shuttle = value["shuttle"]
    if set(shuttle) != {"status", "img_xy", "landing_xy_m_est", "last_shot"}:
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

    def __init__(
        self,
        calibration_path: str | Path,
        fusion: dict[str, Any] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.calibration_path = Path(calibration_path)
        self.homography = load_homography(self.calibration_path)
        # Tunables from config `fusion`; a missing section or key keeps the default.
        identity = (fusion or {}).get("identity") or {}
        rally = (fusion or {}).get("rally") or {}
        shuttle = (fusion or {}).get("shuttle") or {}
        # Window = past-only stand-in for the LESSONS +/-45 frames.
        self._app_window_frames = int(identity.get("app_window_frames", 90))
        # Metres of position cost per unit of L1 appearance distance.
        self._app_weight_m = float(identity.get("app_weight_m", 2.0))
        # Larger than any court distance plus appearance term, so the net line decides side.
        self._side_mismatch_cost = float(identity.get("side_mismatch_cost", 100.0))
        # ponytail: T01 should add `fusion.players: {median_frames: 5, deadband_m: 0.15}` to
        # config/default.yaml; until then these defaults apply. Picked by replaying a 180 s
        # doubles WS log, not tuned against ground truth.
        players = (fusion or {}).get("players") or {}
        self._median_frames = max(1, int(players.get("median_frames", 5)))
        self._deadband_m = float(players.get("deadband_m", 0.15))
        self._kf = ShuttleKalmanFilter(
            predict_ms=float(shuttle.get("predict_ms", 167.0)),
            max_px_per_s=shuttle.get("max_px_per_s"),
            direction_max_gap_ms=float(shuttle.get("direction_max_gap_ms", 500.0)),
            rest_radius_px=float(shuttle.get("rest_radius_px", 8.0)),
            rest_ms=float(shuttle.get("rest_ms", 400.0)),
        )
        self._fsm = RallyFSM(
            homography=self.homography,
            end_gap_ms=float(rally.get("end_gap_ms", 1800.0)),
            rest_landing_ms=float(rally.get("rest_landing_ms", 3000.0)),
        )
        self._shots = ShotDetector(
            match_window_ms=float(rally.get("shot_match_window_ms", 100.0)),
            refractory_ms=float(rally.get("shot_refractory_ms", 200.0)),
        )
        self._workers = {
            name: {"status": "down", "last_seen": 0.0, "fps": None} for name in _WORKERS
        }
        self._latencies_ms: deque[float] = deque(maxlen=256)
        self._events: list[dict[str, Any]] = []
        # Point suggestions awaiting the operator, keyed by event ts.
        self._pending_points: dict[int, dict[str, Any]] = {}
        # Suggestions the operator already answered; a late re-emission must not reopen them.
        self._resolved_points: deque[int] = deque(maxlen=16)
        self._track_to_id: dict[int, int] = {}
        self._mode = "singles"
        self._player_config: dict[int, dict[str, Any]] = {}
        self._players: dict[int, dict[str, Any]] = {}
        self._player_history: dict[int, tuple[int, float, float]] = {}
        # Per slot: (recent court xy window, last counted position, its ts) for dist/speed.
        self._motion: dict[int, tuple[deque[list[float]], list[float], int]] = {}
        self._app_history: dict[int, deque[list[float]]] = {}
        self._score: list[int | None] = [None, None]
        self._games: list[int | None] = [None, None]
        self._server: int | None = None
        self._shuttle: dict[str, Any] = {
            "status": "lost",
            "img_xy": None,
            # Stays null until landing estimation exists; a flying shuttle has no court xy.
            "landing_xy_m_est": None,
            "last_shot": {"by": None, "speed_kmh_est": None, "type": None},
        }
        self._rally = {"active": False, "n": 0, "t_ms": 0, "shots": 0}
        self._ts = 0
        # health.fps / health.dropped. Capture heartbeats reach only the supervisor (stdout),
        # so fusion derives both from the capture frame_id and capture ts that every model
        # worker message carries:
        # - fps = capture frame rate = frame_id advance / capture-ts span over the newest
        #   frames any model worker (players, shuttle) published.
        # - dropped = captured frames the slowest model worker skipped: per worker, the sum
        #   of frame_id gaps in its own stream; the max over workers is reported.
        self._last_frame_ids: dict[str, int] = {}
        self._skipped: dict[str, int] = {"players": 0, "shuttle": 0}
        self._capture_frames: deque[tuple[int, int]] = deque(maxlen=120)

    def _record_latency(self, ts: int) -> None:
        now_ns = time.monotonic_ns()
        if 0 < ts <= now_ns:
            self._latencies_ms.append((now_ns - ts) / 1_000_000)
        self._ts = max(self._ts, ts)

    def _record_frame(self, worker: str, frame_id: int, ts: int) -> None:
        last = self._last_frame_ids.get(worker)
        if last is not None and frame_id > last + 1:
            self._skipped[worker] += frame_id - last - 1
        self._last_frame_ids[worker] = frame_id
        frames = self._capture_frames
        if not frames or frame_id > frames[-1][0]:
            frames.append((frame_id, ts))
        elif ts > frames[-1][1]:  # newer capture time but lower id: capture restarted
            frames.clear()
            frames.append((frame_id, ts))

    def _capture_fps(self) -> float | None:
        if len(self._capture_frames) < 2:
            return None
        (first_id, first_ts), (last_id, last_ts) = self._capture_frames[0], self._capture_frames[-1]
        return (last_id - first_id) * 1e9 / (last_ts - first_ts) if last_ts > first_ts else None

    def ingest(self, payload: str | bytes) -> None:
        message = json.loads(payload)
        kind = message.get("type")
        now = time.monotonic()
        with self._lock:
            if kind == "players":
                decoded = PlayersMessage.from_json(payload)
                self._record_latency(decoded.ts)
                self._record_frame("players", decoded.frame_id, decoded.ts)
                for player in self._players.values():
                    # Keep speed_mps so EMA smoothing survives; position fields go null.
                    player.update(tracked=False, img_xy=None, bbox=None, court_xy_m=None)
                detections = [
                    (detection, self._court_point(detection.foot_px))
                    for detection in decoded.players
                    if detection.track_id is not None
                ]
                # Off-court feet carry no position, so such a track keeps its slot.
                assigned = {
                    detection.track_id: self._track_to_id[detection.track_id]
                    for detection, court_xy in detections
                    if court_xy is None and detection.track_id in self._track_to_id
                }
                free = [s for s in range(self._player_count) if s not in assigned.values()]
                positioned = [(d, xy) for d, xy in detections if xy is not None]
                medians = {slot: self._app_median(slot) for slot in free}
                costs = [
                    [
                        self._match_cost(slot, court_xy, detection.app, medians[slot])
                        for slot in free
                    ]
                    for detection, court_xy in positioned
                ]
                for row, col in _min_cost_pairs(costs):
                    assigned[positioned[row][0].track_id] = free[col]
                for detection, court_xy in detections:
                    free = [s for s in range(self._player_count) if s not in assigned.values()]
                    if court_xy is None and detection.track_id not in assigned and free:
                        assigned[detection.track_id] = free[0]
                self._track_to_id = assigned
                for detection, court_xy in detections:
                    if detection.track_id not in assigned:
                        continue
                    side = None if court_xy is None else int(court_xy[0] > _NET_X_M)
                    player_id = assigned[detection.track_id]
                    apps = self._app_history.setdefault(
                        player_id, deque(maxlen=self._app_window_frames)
                    )
                    # A length change (config edit) restarts the window instead of mixing.
                    if detection.app is not None:
                        if apps and len(apps[0]) != len(detection.app):
                            apps.clear()
                        apps.append(detection.app)
                    previous_player = self._players.get(player_id, {})
                    speed: float | None = None
                    distance: float | None = previous_player.get("dist_m")
                    motion = self._motion.get(player_id)
                    if court_xy is not None and motion is None:
                        self._motion[player_id] = (
                            deque([court_xy], maxlen=self._median_frames),
                            court_xy,
                            decoded.ts,
                        )
                    elif court_xy is not None and motion is not None:
                        window, anchor_xy, anchor_ts = motion
                        window.append(court_xy)
                        # Past-only median drops single-frame foot flicker; the deadband
                        # then ignores what jitter is left, so standing still adds ~0 m.
                        filtered = [statistics.median(axis) for axis in zip(*window, strict=True)]
                        moved = math.dist(filtered, anchor_xy)
                        elapsed_s = (decoded.ts - anchor_ts) / 1_000_000_000
                        if elapsed_s > 0:
                            # Over the time since the last counted step: a stationary
                            # player's speed decays to 0 instead of tracking jitter.
                            measured = moved / elapsed_s
                            old = previous_player.get("speed_mps")
                            speed = measured if old is None else 0.35 * measured + 0.65 * old
                            distance = distance or 0.0
                            if moved >= self._deadband_m:
                                distance += moved
                                self._motion[player_id] = (window, filtered, decoded.ts)
                    # Off-court feet have no court position; keep the last valid one instead.
                    if court_xy is not None:
                        self._player_history[player_id] = (decoded.ts, *court_xy)
                    metadata = self._player_config.get(player_id, {})
                    self._players[player_id] = {
                        "id": player_id,
                        "name": metadata.get("name"),
                        "color": metadata.get("color"),
                        "team": metadata.get("team", player_id % 2),
                        "side": self._slot_side(player_id) if side is None else side,
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
                self._record_frame("shuttle", decoded.frame_id, decoded.ts)
                res = self._kf.update(decoded.visible, decoded.px, decoded.ts)
                self._shuttle.update(status=res.status, img_xy=res.px)
                self._fsm.on_shuttle_update(
                    res.status, self._kf.last_seen_px, decoded.ts, self._kf.rest_px
                )
                if res.direction_changed:
                    audio = self._workers["audio"]
                    audio_ok = (
                        audio["status"] == "ok" and now - audio["last_seen"] <= _HEARTBEAT_TIMEOUT_S
                    )
                    self._on_shot(
                        self._shots.register_direction_change(
                            decoded.ts,
                            px=res.px,
                            players=[p for p in self._players.values() if p["tracked"]],
                            audio_degraded=not audio_ok,
                        )
                    )
                self._sync_rally()
            elif kind == "hit":
                decoded = HitMessage.from_json(payload)
                self._record_latency(decoded.ts)
                self._on_shot(self._shots.register_audio_hit(decoded.ts, decoded.strength))
                self._sync_rally()
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
            self._fsm.homography = self.homography

    def set_players(self, players: list[dict[str, Any]], mode: str = "singles") -> None:
        if mode not in ("singles", "doubles"):
            raise ValueError("mode must be singles or doubles")
        expected = 2 if mode == "singles" else 4
        if len(players) != expected:
            raise ValueError(f"{mode.title()} requires exactly {expected} players")
        player_ids = [int(player["id"]) for player in players]
        if set(player_ids) != set(range(len(players))):
            raise ValueError("Player IDs must be consecutive starting at zero")
        side_counts = [0, 0]
        for player in players:
            side = player.get("side")
            if side not in (0, 1):
                raise ValueError("Every player side must be 0 or 1")
            side_counts[side] += 1
        if any(count not in (1, 2) for count in side_counts):
            raise ValueError("Each side must have one or two players")
        team_counts = [0, 0]
        for player in players:
            team = player.get("team", player.get("side"))
            if team not in (0, 1):
                raise ValueError("Every player team must be 0 or 1")
            team_counts[team] += 1
        if team_counts != [expected // 2] * 2:
            raise ValueError("Each team must have the same number of players")
        with self._lock:
            new_mode = mode
            if new_mode != self._mode:
                self._players.clear()
                self._player_history.clear()
                self._motion.clear()
                self._app_history.clear()
                self._track_to_id.clear()
            self._mode = new_mode
            self._fsm.mode = new_mode
            # Legacy bodies without `team` keep the old meaning: team = starting side.
            self._player_config = {
                int(player["id"]): {"team": player["side"], **player} for player in players
            }
            for player_id, value in self._players.items():
                metadata = self._player_config.get(player_id, {})
                value.update(
                    name=metadata.get("name"),
                    color=metadata.get("color"),
                    team=metadata.get("team", player_id % 2),
                )

    def add_score(self, team: int, delta: int) -> None:
        if team not in (0, 1):
            raise ValueError("team must be 0 or 1")
        with self._lock:
            self._score[team] = (self._score[team] or 0) + delta

    def suggest_point(
        self, event_ts: int, suggested_side: int | None, suggested_team: int | None = None
    ) -> None:
        if suggested_side not in (None, 0, 1) or suggested_team not in (None, 0, 1):
            raise ValueError("suggested side and team must be 0, 1, or null")
        data = {"suggested_side": suggested_side, "suggested_team": suggested_team}
        with self._lock:
            self._pending_points[event_ts] = data
            self._events.append({"ts": event_ts, "kind": "point_suggested", "data": data})

    def confirm_point(self, event_ts: int, accept: bool) -> None:
        with self._lock:
            if event_ts not in self._pending_points:
                raise ValueError("No pending point suggestion for event_ts")
            data = self._pending_points.pop(event_ts)
            self._resolved_points.append(event_ts)
            team = data.get("suggested_team")
            side = data.get("suggested_side")
            if team is None and side is not None:
                # Score is indexed by team; after an end change a side holds the other team.
                team = next(
                    (
                        self._player_config.get(slot, {}).get("team", slot % 2)
                        for slot in range(self._player_count)
                        if self._slot_side(slot) == side
                    ),
                    None,
                )
            if accept and team is not None:
                self._score[team] = (self._score[team] or 0) + 1

    def reset_rally(self) -> None:
        with self._lock:
            self._fsm.reset_to_idle()
            self._sync_rally()
            self._pending_points.clear()

    def swap_ids(self, first: int, second: int) -> None:
        if first not in range(self._player_count) or second not in range(self._player_count):
            raise ValueError(f"player IDs must be between 0 and {self._player_count - 1}")
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
            for slot, value in (
                (first, self._motion.pop(second, None)),
                (second, self._motion.pop(first, None)),
            ):
                if value is not None:
                    self._motion[slot] = value
            # Appearance follows the track too, so the next match keeps the operator's swap.
            self._app_history[first], self._app_history[second] = (
                self._app_history.get(second, deque(maxlen=self._app_window_frames)),
                self._app_history.get(first, deque(maxlen=self._app_window_frames)),
            )
            self._players[first], self._players[second] = (
                self._players.get(second),
                self._players.get(first),
            )
            # Live data follows the track; name, colour and team stay with the id.
            for player_id in (first, second):
                if self._players[player_id] is None:
                    self._players.pop(player_id)
                    continue
                metadata = self._player_config.get(player_id, {})
                self._players[player_id].update(
                    id=player_id,
                    name=metadata.get("name"),
                    color=metadata.get("color"),
                    team=metadata.get("team", player_id % 2),
                )

    def snapshot(self, *, drain_events: bool = True) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            health_workers: dict[str, str] = {}
            for name, worker in self._workers.items():
                if now - worker["last_seen"] > _HEARTBEAT_TIMEOUT_S:
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
            players = [
                self._players.get(index, self._empty_player(index))
                for index in range(self._player_count)
            ]
            result = {
                "v": 1,
                "ts": self._ts,
                "match": {
                    "mode": self._mode,
                    "score": list(self._score),
                    "games": list(self._games),
                    "server": self._server,
                },
                "rally": dict(self._rally),
                "players": players,
                "shuttle": json.loads(json.dumps(self._shuttle)),
                "events": list(events),
                "health": {
                    "fps": self._capture_fps(),
                    "latency_ms_p50": p50,
                    "latency_ms_p99": p99,
                    "dropped": max(self._skipped.values()),
                    "workers": health_workers,
                },
            }
            validate_game_state(result)
            return result

    def _on_shot(self, shot: ShotInfo | None) -> None:
        if shot is not None:
            self._fsm.on_shot(shot)

    def _sync_rally(self) -> None:
        """Copy rally FSM output into the snapshot and queue its new events."""
        self._rally = self._fsm.get_rally_dict()
        # The newest shot's speed needs its return, so show the answered shot: by and
        # speed_kmh_est then always describe the same shot.
        shot = self._fsm.prev_shot
        self._shuttle["landing_xy_m_est"] = self._fsm.landing_xy_m_est
        self._shuttle["last_shot"] = {
            "by": shot.by if shot else None,
            "speed_kmh_est": shot.speed_kmh_est if shot else None,
            "type": shot.type if shot else None,
        }
        for event in self._fsm.drain_events():
            if event["kind"] == "point_suggested":
                # The FSM re-emits with the same ts when the rest point corrects the landing.
                if event["ts"] in self._resolved_points:
                    continue
                self._pending_points[event["ts"]] = event["data"]
            self._events.append(event)

    def _empty_player(self, player_id: int) -> dict[str, Any]:
        metadata = self._player_config.get(player_id, {})
        return {
            "id": player_id,
            "name": metadata.get("name"),
            "color": metadata.get("color"),
            "team": metadata.get("team", player_id % 2),
            "side": metadata.get("side", player_id % 2),
            "tracked": False,
            "img_xy": None,
            "bbox": None,
            "court_xy_m": None,
            "speed_mps": None,
            "dist_m": None,
        }

    @property
    def _player_count(self) -> int:
        return 2 if self._mode == "singles" else 4

    def _slot_side(self, slot: int) -> int:
        # Last side seen from the net line, else the operator's starting side.
        player = self._players.get(slot)
        if player is not None:
            return player["side"]
        return self._player_config.get(slot, {}).get("side", slot % 2)

    def _app_median(self, slot: int) -> list[float] | None:
        # Per-frame appearance is too noisy (LESSONS 4); past frames only, as this is live.
        history = self._app_history.get(slot)
        if not history:
            return None
        return [statistics.median(values) for values in zip(*history, strict=True)]

    def _match_cost(
        self,
        slot: int,
        court_xy: list[float],
        app: list[float] | None,
        median: list[float] | None,
    ) -> float:
        side = int(court_xy[0] > _NET_X_M)
        cost = 0.0 if self._slot_side(slot) == side else self._side_mismatch_cost
        previous = self._player_history.get(slot)
        if previous is not None:
            cost += math.dist(court_xy, previous[1:])
        # Unknown appearance on either end falls back to position only.
        if app is not None and median is not None and len(app) == len(median):
            cost += self._app_weight_m * sum(abs(a - b) for a, b in zip(app, median, strict=True))
        return cost

    def _court_point(self, point_px: list[float]) -> list[float] | None:
        if self.homography is None:
            return None
        point = transform_point(self.homography, point_px)
        lower_y, upper_y = (0.0, 6.10) if self._mode == "doubles" else (0.46, 5.64)
        return point if lower_y <= point[1] <= upper_y else None
