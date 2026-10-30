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
    # [0, 1) = vision only, scaled by racket reach; 1 + strength = vision matched with audio
    evidence: float = 0.0
    px: list[float] | None = None  # shuttle image point at the hit
    candidates: list[dict[str, Any]] | None = None  # tracked players at the hit, for re-attribution
    replaces_ts: int | None = None  # a stronger candidate for the same hit (inside the refractory)


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
        refractory_ms: float = 300.0,
        reach_ratio: float = 0.5,
    ) -> None:
        self.match_window_ns = int(match_window_ms * 1_000_000)
        # Two candidates closer than this are one hit; the stronger is kept.
        self.refractory_ns = int(refractory_ms * 1_000_000)
        # A hit happens within racket reach: shuttle-to-bbox gap <= reach_ratio x bbox height
        # (LESSONS 5). Proposed config key: `fusion.rally.reach_ratio`.
        self.reach_ratio = reach_ratio

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

        A turn outside racket reach of every tracked player (``reach_ratio``) is no
        candidate. In degraded audio mode, visual direction change immediately confirms a
        shot. Otherwise, requires an acoustic hit within <= 100ms.
        """
        gaps = [g for p in players or [] if (g := _reach_gap(px, p)) is not None]
        if gaps and min(gaps) > self.reach_ratio:
            return None  # out of every racket's reach: an apex, jitter or a false detection
        change = _DirectionChange(ts, px, court_xy, players)
        self._recent_dir_changes.append(change)

        if audio_degraded:
            # Vision-only strength: how close the turn is to a racket, below any audio match.
            strength = 0.9 * (1.0 - min(gaps) / self.reach_ratio) if gaps else 0.0
            return self._confirm_shot(change, ts, strength)

        for hit_ts, strength in list(self._recent_hits):
            if abs(ts - hit_ts) <= self.match_window_ns:
                return self._confirm_shot(change, min(ts, hit_ts), 1.0 + strength)

        return None

    @staticmethod
    def attribute_player(
        px: list[float] | tuple[float, float] | None,
        players: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        """Return the tracked player nearest the shuttle image point at hit time.

        Distance is to the player's bbox (0 inside it) divided by the bbox height, so a far
        player standing visually under a near player's racket is not preferred for being
        small; without a bbox it is the pixel distance to the foot point. Inside a rally the
        rally FSM narrows ``players`` to the side alternation says is hitting.
        """
        if px is None or not players:
            return None

        best: dict[str, Any] | None = None
        min_dist = float("inf")
        for player in players:
            if player.get("id") is None:
                continue
            dist = _reach_gap(px, player)
            if dist is None and player.get("img_xy") is not None:
                dist = math.hypot(px[0] - player["img_xy"][0], px[1] - player["img_xy"][1])
            if dist is not None and dist < min_dist:
                min_dist = dist
                best = player
        return best

    def _confirm_shot(
        self, change: _DirectionChange, shot_ts: int, evidence: float
    ) -> ShotInfo | None:
        replaces_ts: int | None = None
        if shot_ts - self._last_shot_ts < self.refractory_ns:
            # Too close to be a second hit: keep whichever candidate has stronger evidence.
            last = self.last_shot
            if last is None or evidence <= last.evidence:
                return None
            replaces_ts = last.ts

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
            px=change.px,
            candidates=change.players,
            replaces_ts=replaces_ts,
        )
        self.last_shot = shot
        return shot


def _reach_gap(
    px: list[float] | tuple[float, float] | None, player: dict[str, Any]
) -> float | None:
    """Shuttle-to-bbox distance in bbox heights (0 inside it); None without px or bbox."""
    bbox = player.get("bbox")
    if px is None or bbox is None:
        return None
    dx = max(bbox[0] - px[0], 0.0, px[0] - bbox[2])
    dy = max(bbox[1] - px[1], 0.0, px[1] - bbox[3])
    return math.hypot(dx, dy) / max(1.0, bbox[3] - bbox[1])
