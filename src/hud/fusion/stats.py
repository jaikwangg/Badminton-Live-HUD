"""Badminton shot counting, speed estimation, and stroke attribution.

According to CONTRACTS and T08:
- Shots are counted by matching audio hit and visual direction change within <= 100ms.
- If audio is degraded or unavailable, shots are counted purely from visual direction change.
- Shuttlecock speed (speed_kmh_est) is calculated over the first 3-5 frames after a stroke.
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


class SpeedEstimator:
    """Estimate shuttlecock launch speed from court coordinates across 3-5 consecutive frames."""

    def __init__(self, min_frames: int = 3, max_frames: int = 5) -> None:
        self.min_frames = min_frames
        self.max_frames = max_frames
        self._samples: list[tuple[int, float, float]] = []
        self._active = False

    def start_sampling(self, ts: int, court_xy: list[float] | tuple[float, float]) -> None:
        """Begin collecting court positions following a confirmed stroke."""
        self._samples = [(ts, float(court_xy[0]), float(court_xy[1]))]
        self._active = True

    def add_sample(
        self, ts: int, court_xy: list[float] | tuple[float, float] | None
    ) -> float | None:
        """Add a post-stroke court sample and compute speed once 3-5 frames are gathered."""
        if not self._active or court_xy is None:
            return None

        self._samples.append((ts, float(court_xy[0]), float(court_xy[1])))

        if len(self._samples) >= self.max_frames:
            speed = self._calculate_speed()
            self._active = False
            return speed

        return None

    def finalize(self) -> float | None:
        """Force speed calculation if at least min_frames were recorded."""
        if not self._active:
            return None
        self._active = False
        if len(self._samples) >= self.min_frames:
            return self._calculate_speed()
        return None

    def _calculate_speed(self) -> float | None:
        if len(self._samples) < self.min_frames:
            return None

        total_distance = 0.0
        for i in range(len(self._samples) - 1):
            _, x1, y1 = self._samples[i]
            _, x2, y2 = self._samples[i + 1]
            total_distance += math.hypot(x2 - x1, y2 - y1)

        dt = (self._samples[-1][0] - self._samples[0][0]) / 1_000_000_000.0
        if dt <= 0.0 or total_distance <= 0.0:
            return None

        speed_mps = total_distance / dt
        speed_kmh = speed_mps * 3.6
        return round(speed_kmh, 1)


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
        self._recent_dir_changes: deque[tuple[int, list[float] | None]] = deque(maxlen=16)

        self._last_shot_ts: int = -self.refractory_ns
        self.speed_estimator = SpeedEstimator(min_frames=3, max_frames=5)
        self.last_shot: ShotInfo | None = None

    def reset(self) -> None:
        self._recent_hits.clear()
        self._recent_dir_changes.clear()
        self._last_shot_ts = -self.refractory_ns
        self.speed_estimator = SpeedEstimator(min_frames=3, max_frames=5)
        self.last_shot = None

    def register_audio_hit(self, ts: int, strength: float = 0.5) -> ShotInfo | None:
        """Register an acoustic hit event. Returns ShotInfo if matching visual change found."""
        self._recent_hits.append((ts, strength))

        # Check if there is an existing visual direction change within 100ms
        for dir_ts, court_xy in list(self._recent_dir_changes):
            if abs(ts - dir_ts) <= self.match_window_ns:
                return self._confirm_shot(shot_ts=min(ts, dir_ts), court_xy=court_xy)

        return None

    def register_direction_change(
        self,
        ts: int,
        court_xy: list[float] | None = None,
        *,
        audio_degraded: bool = False,
        players: list[dict[str, Any]] | None = None,
    ) -> ShotInfo | None:
        """Register a visual direction change.

        In degraded audio mode, visual direction change immediately confirms a shot.
        Otherwise, requires an acoustic hit within <= 100ms.
        """
        self._recent_dir_changes.append((ts, court_xy))

        if audio_degraded:
            # Degraded mode: count purely from visual direction change
            return self._confirm_shot(shot_ts=ts, court_xy=court_xy, players=players)

        # Standard mode: check for matching acoustic hit within 100ms
        for hit_ts, _ in list(self._recent_hits):
            if abs(ts - hit_ts) <= self.match_window_ns:
                return self._confirm_shot(
                    shot_ts=min(ts, hit_ts), court_xy=court_xy, players=players
                )

        return None

    def attribute_player(
        self,
        court_xy: list[float] | tuple[float, float] | None,
        players: list[dict[str, Any]] | None,
    ) -> int | None:
        """Determine which player executed the stroke based on proximity to shuttle court_xy."""
        if court_xy is None or not players:
            return None

        sx, sy = float(court_xy[0]), float(court_xy[1])
        closest_id: int | None = None
        min_dist = float("inf")

        for player in players:
            player_id = player.get("id")
            p_court = player.get("court_xy_m")
            if player_id is not None and p_court is not None:
                dist = math.hypot(sx - float(p_court[0]), sy - float(p_court[1]))
                if dist < min_dist:
                    min_dist = dist
                    closest_id = int(player_id)

        return closest_id

    def _confirm_shot(
        self,
        shot_ts: int,
        court_xy: list[float] | None = None,
        players: list[dict[str, Any]] | None = None,
    ) -> ShotInfo | None:
        if shot_ts - self._last_shot_ts < self.refractory_ns:
            return None

        self._last_shot_ts = shot_ts
        player_id = self.attribute_player(court_xy, players)

        if court_xy is not None:
            self.speed_estimator.start_sampling(shot_ts, court_xy)

        shot = ShotInfo(
            ts=shot_ts,
            by=player_id,
            speed_kmh_est=None,
            court_xy_m=court_xy,
        )
        self.last_shot = shot
        return shot

    def update_post_shot_court_sample(self, ts: int, court_xy: list[float] | None) -> float | None:
        """Forward post-shot shuttlecourt positions to speed estimator."""
        speed = self.speed_estimator.add_sample(ts, court_xy)
        if speed is not None and self.last_shot is not None:
            self.last_shot.speed_kmh_est = speed
        return speed
