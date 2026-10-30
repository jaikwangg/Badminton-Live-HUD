"""Shuttlecock Kalman filter for 2D position/velocity tracking and state estimation.

According to CONTRACTS and T08:
- Missed for up to 10 frames (~167 ms at a nominal 60 fps): status is 'predicted'.
- Missed for longer: status transitions to 'lost' with None coordinates.

Both are measured in time from the last accepted detection, not in messages, because the
shuttle worker publishes only inferred frames and their spacing follows inference speed.
"""

from __future__ import annotations

import math
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
    ) -> None:
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
        self._prev_velocity = (0.0, 0.0)
        self._last_observation: tuple[float, float] | None = None
        self.last_seen_ts: int | None = None
        self.status: ShuttleStatus = "lost"

    @property
    def last_seen_px(self) -> list[float] | None:
        """Last accepted image detection; the landing estimate is taken from it."""
        return list(self._last_observation) if self._last_observation is not None else None

    def reset(self) -> None:
        """Clear filter state and mark shuttle as lost."""
        self._x = np.zeros(4, dtype=np.float64)
        self._p = np.diag([50.0, 50.0, 1_000_000.0, 1_000_000.0]).astype(np.float64)
        self._initialized = False
        self._last_ts = None
        self._prev_velocity = (0.0, 0.0)
        self._last_observation = None
        self.last_seen_ts = None
        self._jump_frames = 0
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

    def _detect_direction_change(self, curr_vel: tuple[float, float]) -> bool:
        vx1, vy1 = self._prev_velocity
        vx2, vy2 = curr_vel
        speed1 = math.hypot(vx1, vy1)
        speed2 = math.hypot(vx2, vy2)

        # Minimum speed threshold to consider direction reliable (e.g. 50 px/s)
        if speed1 < 50.0 or speed2 < 50.0:
            return False

        # Cosine similarity between velocity vectors
        dot = (vx1 * vx2 + vy1 * vy2) / (speed1 * speed2)
        dot = max(-1.0, min(1.0, dot))
        angle = math.acos(dot)

        # Significant deflection angle indicates a stroke / hit direction change
        return angle >= self.direction_change_rad

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

            if not self._initialized or self.status == "lost":
                # At low message rates one miss already means lost; keep the direction
                # history across a short gap instead of needing 3 fresh detections.
                bridged = (
                    self._last_observation is not None
                    and self.last_seen_ts is not None
                    and 0 < ts - self.last_seen_ts <= self.direction_max_gap_ns
                )
                velocity = (0.0, 0.0)
                dir_changed = False
                if bridged:
                    gap_s = (ts - self.last_seen_ts) / 1e9
                    last_x, last_y = self._last_observation
                    velocity = ((zx - last_x) / gap_s, (zy - last_y) / gap_s)
                    dir_changed = self._detect_direction_change(velocity)
                self._x = np.array([zx, zy, *velocity], dtype=np.float64)
                self._p = np.diag([10.0, 10.0, 1_000_000.0, 1_000_000.0]).astype(np.float64)
                self._initialized = True
                self.status = "tracked"
                self._prev_velocity = velocity
                self._last_observation = (zx, zy)
                self.last_seen_ts = ts
                self._jump_frames = 0
                return ShuttleTrackResult(
                    status="tracked",
                    px=[zx, zy],
                    velocity=velocity,
                    speed_px_s=math.hypot(*velocity),
                    direction_changed=dir_changed,
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
            dir_changed = self._detect_direction_change(observed_velocity)
            self._prev_velocity = observed_velocity
            self._last_observation = (zx, zy)
            self.last_seen_ts = ts
            speed = math.hypot(curr_vel[0], curr_vel[1])

            return ShuttleTrackResult(
                status="tracked",
                px=[float(self._x[0]), float(self._x[1])],
                velocity=curr_vel,
                speed_px_s=speed,
                direction_changed=dir_changed,
            )

        # Shuttle is not visible in this frame
        if not self._initialized:
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
