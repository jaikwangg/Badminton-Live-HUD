"""Badminton shot counting, speed estimation, and stroke attribution.

According to CONTRACTS and T08:
- Shots are counted by matching audio hit and visual direction change within <= 100ms.
- If audio is degraded or unavailable, shots are counted purely from visual direction change.
- Shuttle speed (speed_kmh_est) uses the drag model of LESSONS 7 between a shot and its
  return, so a shot's speed is known only once the next shot happens.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ShotInfo:
    ts: int
    by: int | None
    speed_kmh_est: float | None
    type: str | None = None
    court_xy_m: list[float] | None = None
    hitter_xy_m: list[float] | None = None  # hitter's court position at the hit
    side: int | None = None  # court side of the hitter, used to force alternation
    team: int | None = None
    evidence: float = 0.0  # 0 = vision only, 1 + strength = vision matched with audio


@dataclass(slots=True)
class _DirectionChange:
    ts: int
    px: list[float] | None
    court_xy: list[float] | None
    players: list[dict[str, Any]] | None


# ponytail: speed guards are module constants; move them into config when T01 owns them.
GRAVITY_MPS2 = 9.81
SHUTTLE_TERMINAL_MPS = 6.8
DRAG_K_PER_M = GRAVITY_MPS2 / SHUTTLE_TERMINAL_MPS**2
# T under 0.4 s gives absurd values (LESSONS 7: 0.30 s -> 325 km/h); over 3 s is not one flight.
SPEED_MIN_DT_S = 0.4
SPEED_MAX_DT_S = 3.0
# Shots closer than 3 m (net play) have no usable estimate (LESSONS 7).
SPEED_MIN_DIST_M = 3.0


def drag_speed_kmh_est(
    hitter_xy_m: list[float] | None, returner_xy_m: list[float] | None, dt_ns: int
) -> float | None:
    """Launch speed v0 = (e^(kD) - 1) / (kT) of a shot, from the hitter and returner.

    D is the court ground distance between the two players and T the time between the
    two shots. Returns None when a position is unknown or D/T is outside sane bounds.
    """
    if hitter_xy_m is None or returner_xy_m is None:
        return None
    dt_s = dt_ns / 1_000_000_000.0
    if not SPEED_MIN_DT_S <= dt_s <= SPEED_MAX_DT_S:
        return None
    dist_m = math.hypot(returner_xy_m[0] - hitter_xy_m[0], returner_xy_m[1] - hitter_xy_m[1])
    if dist_m < SPEED_MIN_DIST_M:
        return None
    v0_mps = math.expm1(DRAG_K_PER_M * dist_m) / (DRAG_K_PER_M * dt_s)
    return round(v0_mps * 3.6, 1)


class ShotDetector:
    """Correlate visual shuttle direction reversals and acoustic hits into confirmed shots."""

    def __init__(
        self,
        match_window_ms: float = 100.0,
        refractory_ms: float = 200.0,
    ) -> None:
        self.match_window_ns = int(match_window_ms * 1_000_000)
        self.refractory_ns = int(refractory_ms * 1_000_000)

        self._recent_hits: deque[tuple[int, float]] = deque(maxlen=16)
        self._recent_dir_changes: deque[_DirectionChange] = deque(maxlen=16)

        self._last_shot_ts: int = -self.refractory_ns
        self.last_shot: ShotInfo | None = None

    def reset(self) -> None:
        self._recent_hits.clear()
        self._recent_dir_changes.clear()
        self._last_shot_ts = -self.refractory_ns
        self.last_shot = None

    def register_audio_hit(self, ts: int, strength: float = 0.5) -> ShotInfo | None:
        """Register an acoustic hit event. Returns ShotInfo if matching visual change found."""
        self._recent_hits.append((ts, strength))

        for change in list(self._recent_dir_changes):
            if abs(ts - change.ts) <= self.match_window_ns:
                return self._confirm_shot(change, min(ts, change.ts), 1.0 + strength)

        return None

    def register_direction_change(
        self,
        ts: int,
        court_xy: list[float] | None = None,
        *,
        px: list[float] | None = None,
        audio_degraded: bool = False,
        players: list[dict[str, Any]] | None = None,
    ) -> ShotInfo | None:
        """Register a visual direction change at shuttle image point ``px``.

        In degraded audio mode, visual direction change immediately confirms a shot.
        Otherwise, requires an acoustic hit within <= 100ms.
        """
        change = _DirectionChange(ts, px, court_xy, players)
        self._recent_dir_changes.append(change)

        if audio_degraded:
            return self._confirm_shot(change, ts, 0.0)

        for hit_ts, strength in list(self._recent_hits):
            if abs(ts - hit_ts) <= self.match_window_ns:
                return self._confirm_shot(change, min(ts, hit_ts), 1.0 + strength)

        return None

    def attribute_player(
        self,
        px: list[float] | tuple[float, float] | None,
        players: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        """Return the tracked player nearest the shuttle image point at hit time.

        Distance is to the player's bbox (0 inside it), else to the foot point. The
        nearest player's side is the shot side, so in doubles this is also the player
        on that side nearest the shuttle.
        """
        if px is None or not players:
            return None

        sx, sy = float(px[0]), float(px[1])
        best: dict[str, Any] | None = None
        min_dist = float("inf")
        for player in players:
            if player.get("id") is None:
                continue
            bbox = player.get("bbox")
            if bbox is not None:
                dx = max(bbox[0] - sx, 0.0, sx - bbox[2])
                dy = max(bbox[1] - sy, 0.0, sy - bbox[3])
            elif player.get("img_xy") is not None:
                dx, dy = sx - player["img_xy"][0], sy - player["img_xy"][1]
            else:
                continue
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                best = player
        return best

    def _confirm_shot(
        self, change: _DirectionChange, shot_ts: int, evidence: float
    ) -> ShotInfo | None:
        if shot_ts - self._last_shot_ts < self.refractory_ns:
            return None

        self._last_shot_ts = shot_ts
        hitter = self.attribute_player(change.px, change.players)

        shot = ShotInfo(
            ts=shot_ts,
            by=int(hitter["id"]) if hitter else None,
            speed_kmh_est=None,
            court_xy_m=change.court_xy,
            hitter_xy_m=hitter.get("court_xy_m") if hitter else None,
            side=hitter.get("side") if hitter else None,
            team=hitter.get("team") if hitter else None,
            evidence=evidence,
        )
        self.last_shot = shot
        return shot
