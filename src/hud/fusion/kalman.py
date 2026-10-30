"""Shuttlecock Kalman filter for 2D position/velocity tracking and state estimation.

According to CONTRACTS and T08:
- Missed for up to 10 frames (~167 ms at a nominal 60 fps): status is 'predicted'.
- Missed for longer: status transitions to 'lost' with None coordinates.

Both are measured in time from the last accepted detection, not in messages, because the
shuttle worker publishes only inferred frames and their spacing follows inference speed.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

ShuttleStatus = Literal["tracked", "predicted", "lost"]


@dataclass(slots=True)
class ShuttleTrackResult:
    status: ShuttleStatus
    px: list[float] | None
    velocity: tuple[float, float]
    speed_px_s: float
    direction_changed: bool
    # The confirmed turn lies turn_points accepted points back: its vertex (nearest the
    # racket) and the ts of the first point after it. Set only when direction_changed.
    turn_ts: int | None = None
    turn_px: list[float] | None = None


class ShuttleKalmanFilter:
    """Constant-velocity 2D Kalman filter tracking shuttlecock image coordinates.

    State vector: [x, y, vx, vy]^T
    Measurement:  [x, y]^T
    """

    def __init__(
        self,
        predict_ms: float = 167.0,
        process_noise_std: float = 120.0,
        measurement_noise_std: float = 4.0,
        direction_change_deg: float = 55.0,
        max_px_per_s: float | None = None,
        max_jump_frames: int = 3,
        direction_max_gap_ms: float = 500.0,
        rest_radius_px: float = 8.0,
        rest_ms: float = 400.0,
        turn_points: int = 3,
    ) -> None:
        # A reversal counts only when turn_points accepted points before and after the vertex
        # each run straight: a single-point turn test fires on apexes, jitter and false
        # detections (accuracy run 1: 137 changes in rallies for 45 shots).
        # Proposed config key: `fusion.shuttle.turn_points`.
        self.turn_points = max(1, turn_points)
        self._turn_history: deque[tuple[int, float, float]] = deque(maxlen=2 * self.turn_points + 1)
        # A shuttle whose detections stay within rest_radius_px of one point for rest_ms is
        # lying on the floor (or held): not in play. Reported as 'lost' (no 'resting' status in
        # the contract), it neither counts as flight nor keeps the rally alive.
        # ponytail: proposed config keys `fusion.shuttle.rest_radius_px` / `rest_ms` for T01;
        # 8 px / 400 ms is from one 4 Hz CPU clip, re-check at GPU rates.
        self.rest_radius_px = rest_radius_px
        self.rest_ns = int(rest_ms * 1_000_000)
        self._rest_anchor: tuple[float, float] | None = None
        self._rest_since_ts = 0
        self.resting = False
        # Direction memory survives a loss up to this gap (a hit hidden by the hitter's body,
        # or one missed message at ~4 Hz). Shot-to-shot is rarely under 0.5 s, so such a gap
        # hides at most one hit and the chord across it still shows the reversal.
        # Proposed config key: `fusion.shuttle.direction_max_gap_ms`; re-check on GPU-rate clips.
        self.direction_max_gap_ns = int(direction_max_gap_ms * 1_000_000)
        # max_px_per_s comes from config `shuttle.max_px_per_s`; None disables the jump gate.
        self.max_px_per_s = max_px_per_s
        self.max_jump_frames = max_jump_frames
        self._jump_frames = 0
        # Proposed config key for T01: `fusion.shuttle.predict_ms` (10 frames at 60 fps).
        self.predict_ns = int(predict_ms * 1_000_000)
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.direction_change_rad = math.radians(direction_change_deg)

        # State vector [x, y, vx, vy]
        self._x = np.zeros(4, dtype=np.float64)
        # Position begins uncertain and velocity is initially unknown.
        self._p = np.diag([50.0, 50.0, 1_000_000.0, 1_000_000.0]).astype(np.float64)

        # Measurement matrix H
        self._h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

        # Measurement noise covariance R
        self._r = np.eye(2, dtype=np.float64) * (measurement_noise_std**2)

        self._initialized = False
        self._last_ts: int | None = None
        self._last_observation: tuple[float, float] | None = None
        self.last_seen_ts: int | None = None
        self.status: ShuttleStatus = "lost"

    @property
    def last_seen_px(self) -> list[float] | None:
        """Last accepted image detection; the landing estimate is taken from it."""
        return list(self._last_observation) if self._last_observation is not None else None

    @property
    def rest_px(self) -> list[float] | None:
        """Image point where the shuttle lies at rest, None unless it is resting now."""
        return list(self._rest_anchor) if self.resting and self._rest_anchor else None

    def reset(self) -> None:
        """Clear filter state and mark shuttle as lost."""
        self._x = np.zeros(4, dtype=np.float64)
        self._p = np.diag([50.0, 50.0, 1_000_000.0, 1_000_000.0]).astype(np.float64)
        self._initialized = False
        self._last_ts = None
        self._turn_history.clear()
        self._last_observation = None
        self.last_seen_ts = None
        self._jump_frames = 0
        self._rest_anchor = None
        self.resting = False
        self.status = "lost"

    def _get_dt(self, ts: int) -> float:
        if self._last_ts is None or ts <= self._last_ts:
            dt = 1.0 / 60.0
        else:
            # Real spacing: messages arrive at inference rate, not at the camera frame rate.
            dt = (ts - self._last_ts) / 1_000_000_000.0
        self._last_ts = ts
        return dt

    def _build_f(self, dt: float) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _build_q(self, dt: float) -> np.ndarray:
        # Discrete constant white noise acceleration model
        q_var = self.process_noise_std**2
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        return np.array(
            [
                [dt4 * q_var, 0.0, dt3 * q_var, 0.0],
                [0.0, dt4 * q_var, 0.0, dt3 * q_var],
                [dt3 * q_var, 0.0, dt2 * q_var, 0.0],
                [0.0, dt3 * q_var, 0.0, dt2 * q_var],
            ],
            dtype=np.float64,
        )

    def _predict(self, dt: float) -> None:
        f = self._build_f(dt)
        q = self._build_q(dt)
        self._x = f @ self._x
        self._p = f @ self._p @ f.T + q

    def _update(self, z: np.ndarray) -> None:
        y = z - (self._h @ self._x)
        s = self._h @ self._p @ self._h.T + self._r
        k = self._p @ self._h.T @ np.linalg.inv(s)
        self._x = self._x + k @ y
        i = np.eye(4, dtype=np.float64)
        self._p = (i - k @ self._h) @ self._p

    def _add_turn_point(self, ts: int, x: float, y: float) -> tuple[int, list[float]] | None:
        """Add an accepted point; return the turn (ts, vertex px) once a reversal is confirmed.

        Velocities are chords over the real capture times, so the test is the same at
        30 and 60 fps (and across a bridged gap at low inference rates).
        """
        history = self._turn_history
        if history and ts - history[-1][0] > self.direction_max_gap_ns:
            history.clear()
        history.append((ts, x, y))
        k = self.turn_points
        if len(history) < 2 * k + 1:
            return None
        points = list(history)
        before, after = points[: k + 1], points[k:]
        v1, v2 = _chord(before), _chord(after)
        if v1 is None or v2 is None or not (_straight(before, v1) and _straight(after, v2)):
            return None
        speed1, speed2 = math.hypot(*v1), math.hypot(*v2)
        # Minimum speed for a reliable direction (50 px/s).
        if speed1 < 50.0 or speed2 < 50.0:
            return None
        cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (speed1 * speed2)
        if math.acos(max(-1.0, min(1.0, cos))) < self.direction_change_rad:
            return None
        # The vertex starts the next segment, so the same turn is never reported twice.
        for _ in range(k):
            history.popleft()
        _, vx, vy = points[k]
        # The hit lies between the vertex and the next point; that point's ts matches the
        # timing of a single-step test, which audio matching was built around.
        return points[k + 1][0], [vx, vy]

    def update(
        self,
        visible: bool,
        px: list[float] | tuple[float, float] | None,
        ts: int,
    ) -> ShuttleTrackResult:
        """Step the filter with new frame observation.

        Args:
            visible: Whether shuttlecock was detected by vision model.
            px: Detected image [x, y] coordinates.
            ts: Monotonic frame capture timestamp in nanoseconds.

        Returns:
            ShuttleTrackResult with status, coordinates, velocity, and direction change.
        """
        dt = self._get_dt(ts)

        if visible and px is not None:
            zx, zy = float(px[0]), float(px[1])
            z = np.array([zx, zy], dtype=np.float64)

            anchor = self._rest_anchor
            if anchor is None or math.hypot(zx - anchor[0], zy - anchor[1]) > self.rest_radius_px:
                # Fixed anchor: a slowly flying shuttle drifts out of the radius and restarts it.
                self._rest_anchor = (zx, zy)
                self._rest_since_ts = ts
                self.resting = False
            elif ts - self._rest_since_ts >= self.rest_ns:
                self.resting = True
            if self.resting:
                # Filter state is left untouched, so the next real flight starts fresh.
                self._turn_history.clear()
                self.status = "lost"
                return ShuttleTrackResult(
                    status="lost",
                    px=None,
                    velocity=(0.0, 0.0),
                    speed_px_s=0.0,
                    direction_changed=False,
                )

            if not self._initialized or self.status == "lost":
                # At low message rates one miss already means lost; keep the direction
                # history across a short gap instead of needing 3 fresh detections.
                bridged = (
                    self._last_observation is not None
                    and self.last_seen_ts is not None
                    and 0 < ts - self.last_seen_ts <= self.direction_max_gap_ns
                )
                velocity = (0.0, 0.0)
                if bridged:
                    gap_s = (ts - self.last_seen_ts) / 1e9
                    last_x, last_y = self._last_observation
                    velocity = ((zx - last_x) / gap_s, (zy - last_y) / gap_s)
                turn = self._add_turn_point(ts, zx, zy)
                self._x = np.array([zx, zy, *velocity], dtype=np.float64)
                self._p = np.diag([10.0, 10.0, 1_000_000.0, 1_000_000.0]).astype(np.float64)
                self._initialized = True
                self.status = "tracked"
                self._last_observation = (zx, zy)
                self.last_seen_ts = ts
                self._jump_frames = 0
                return ShuttleTrackResult(
                    status="tracked",
                    px=[zx, zy],
                    velocity=velocity,
                    speed_px_s=math.hypot(*velocity),
                    direction_changed=turn is not None,
                    turn_ts=turn[0] if turn else None,
                    turn_px=turn[1] if turn else None,
                )

            self._predict(dt)
            last_x, last_y = self._last_observation or (zx, zy)
            # Velocity over the real gap since the last accepted detection, not one frame.
            obs_dt = (
                (ts - self.last_seen_ts) / 1e9
                if self.last_seen_ts and ts > self.last_seen_ts
                else dt
            )
            observed_velocity = ((zx - last_x) / obs_dt, (zy - last_y) / obs_dt)

            if (
                self.max_px_per_s is not None
                and math.hypot(*observed_velocity) > self.max_px_per_s
                and self._jump_frames < self.max_jump_frames
            ):
                # Implausible jump: trust the prediction, but only for a few frames so a
                # genuine new trajectory is accepted and one bad frame never wipes the track.
                self._jump_frames += 1
                self.status = "predicted"
                curr_vel = (float(self._x[2]), float(self._x[3]))
                return ShuttleTrackResult(
                    status="predicted",
                    px=[float(self._x[0]), float(self._x[1])],
                    velocity=curr_vel,
                    speed_px_s=math.hypot(*curr_vel),
                    direction_changed=False,
                )

            self._jump_frames = 0
            self._update(z)
            self.status = "tracked"

            curr_vel = (float(self._x[2]), float(self._x[3]))
            turn = self._add_turn_point(ts, zx, zy)
            self._last_observation = (zx, zy)
            self.last_seen_ts = ts
            speed = math.hypot(curr_vel[0], curr_vel[1])

            return ShuttleTrackResult(
                status="tracked",
                px=[float(self._x[0]), float(self._x[1])],
                velocity=curr_vel,
                speed_px_s=speed,
                direction_changed=turn is not None,
                turn_ts=turn[0] if turn else None,
                turn_px=turn[1] if turn else None,
            )

        # Shuttle is not visible in this frame
        if not self._initialized or self.resting:
            self.status = "lost"
            return ShuttleTrackResult(
                status="lost",
                px=None,
                velocity=(0.0, 0.0),
                speed_px_s=0.0,
                direction_changed=False,
            )

        self._predict(dt)

        curr_vel = (float(self._x[2]), float(self._x[3]))
        speed = math.hypot(curr_vel[0], curr_vel[1])

        if self.last_seen_ts is not None and ts - self.last_seen_ts <= self.predict_ns:
            self.status = "predicted"
            # In predicted state, direction change from a missing measurement is not confirmed
            return ShuttleTrackResult(
                status="predicted",
                px=[float(self._x[0]), float(self._x[1])],
                velocity=curr_vel,
                speed_px_s=speed,
                direction_changed=False,
            )

        # Unseen for longer than predict_ms -> lost
        self.status = "lost"
        return ShuttleTrackResult(
            status="lost",
            px=None,
            velocity=curr_vel,
            speed_px_s=speed,
            direction_changed=False,
        )


def _chord(points: list[tuple[int, float, float]]) -> tuple[float, float] | None:
    """Mean velocity (px/s) from the first to the last (ts, x, y) point."""
    dt_s = (points[-1][0] - points[0][0]) / 1e9
    if dt_s <= 0:
        return None
    return ((points[-1][1] - points[0][1]) / dt_s, (points[-1][2] - points[0][2]) / dt_s)


def _straight(points: list[tuple[int, float, float]], chord: tuple[float, float]) -> bool:
    """Every step moves along the chord: jitter and single outliers break the run."""
    return all(
        (b[1] - a[1]) * chord[0] + (b[2] - a[2]) * chord[1] > 0
        for a, b in zip(points, points[1:], strict=False)
    )
