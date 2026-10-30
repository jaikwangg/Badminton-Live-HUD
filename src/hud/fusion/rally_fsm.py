"""Rally finite-state machine (IDLE -> SERVE -> RALLY -> END) and point recommendation.

According to CONTRACTS and T08:
- Computes rally active state, rally index (n), elapsed duration (t_ms), and shots count.
- At rally end, estimates landing point and emits 'point_suggested' event.
- STRICT RULE: Never modifies match score automatically; requires operator confirmation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from hud.fusion.stats import ShotInfo


class RallyState(StrEnum):
    IDLE = "IDLE"
    SERVE = "SERVE"
    RALLY = "RALLY"
    END = "END"


class RallyFSM:
    """Manage rally lifecycle, state transitions, duration timing, and point recommendations."""

    # Singles court boundary constants (CONTRACTS.md)
    COURT_LENGTH_M = 13.40
    COURT_WIDTH_M = 6.10
    COURT_NET_X_M = 6.70
    SINGLES_Y_MIN_M = 0.46
    SINGLES_Y_MAX_M = 5.64

    def __init__(
        self,
        mode: str = "singles",
        player_sides: dict[int, int] | None = None,
    ) -> None:
        if mode not in {"singles", "doubles"}:
            raise ValueError("mode must be 'singles' or 'doubles'")
        if player_sides is not None and any(side not in (0, 1) for side in player_sides.values()):
            raise ValueError("player sides must be 0 or 1")
        self.mode = mode
        self.player_sides = dict(player_sides or {})
        self.state: RallyState = RallyState.IDLE
        self.rally_n: int = 0
        self.shots_count: int = 0
        self.start_ts: int = 0
        self.end_ts: int = 0
        self.current_t_ms: int = 0

        self.last_shuttle_status: str = "lost"
        self.last_court_xy: list[float] | None = None
        self.last_shot: ShotInfo | None = None

        self._pending_events: list[dict[str, Any]] = []

    @property
    def is_active(self) -> bool:
        return self.state in (RallyState.SERVE, RallyState.RALLY)

    def get_rally_dict(self) -> dict[str, Any]:
        """Return Contract v1 rally dictionary."""
        return {
            "active": self.is_active,
            "n": self.rally_n,
            "t_ms": self.current_t_ms,
            "shots": self.shots_count,
        }

    def drain_events(self) -> list[dict[str, Any]]:
        """Retrieve and clear accumulated events."""
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    def start_serve(self, ts: int) -> None:
        """Transition IDLE -> SERVE."""
        if self.state in (RallyState.IDLE, RallyState.END):
            self.state = RallyState.SERVE
            self.rally_n += 1
            self.shots_count = 0
            self.start_ts = ts
            self.end_ts = 0
            self.current_t_ms = 0
            self.last_court_xy = None
            self.last_shot = None
            self._pending_events.append(
                {"ts": ts, "kind": "rally_start", "data": {"n": self.rally_n}}
            )

    def start_rally(self, ts: int) -> None:
        """Transition SERVE/IDLE -> RALLY."""
        if self.state == RallyState.IDLE:
            self.start_serve(ts)
        self.state = RallyState.RALLY

    def on_shot(self, shot: ShotInfo) -> None:
        """Record a confirmed shot during the rally."""
        if self.state == RallyState.IDLE:
            self.start_serve(shot.ts)
            self.state = RallyState.RALLY
        elif self.state == RallyState.SERVE:
            self.state = RallyState.RALLY

        if self.state == RallyState.RALLY:
            self.shots_count += 1
            self.last_shot = shot
            if shot.court_xy_m is not None:
                self.last_court_xy = shot.court_xy_m

            self._pending_events.append(
                {
                    "ts": shot.ts,
                    "kind": "shot",
                    "data": {
                        "by": shot.by,
                        "speed_kmh_est": shot.speed_kmh_est,
                        "shots": self.shots_count,
                    },
                }
            )

    def on_shuttle_update(
        self,
        status: str,
        court_xy_m: list[float] | None,
        ts: int,
    ) -> None:
        """Receive updated shuttle status from Kalman filter."""
        self.last_shuttle_status = status
        if court_xy_m is not None:
            self.last_court_xy = court_xy_m

        if self.is_active:
            if self.start_ts > 0 and ts >= self.start_ts:
                self.current_t_ms = (ts - self.start_ts) // 1_000_000

            # If shuttle is lost during an active rally after shots were fired, conclude the rally
            if status == "lost" and self.shots_count >= 1:
                self.end_rally(ts)

    def end_rally(self, ts: int) -> None:
        """Transition RALLY/SERVE -> END."""
        if not self.is_active:
            return

        self.state = RallyState.END
        self.end_ts = ts
        if self.start_ts > 0 and ts >= self.start_ts:
            self.current_t_ms = (ts - self.start_ts) // 1_000_000

        # 1. Emit rally_end event
        self._pending_events.append(
            {
                "ts": ts,
                "kind": "rally_end",
                "data": {
                    "n": self.rally_n,
                    "t_ms": self.current_t_ms,
                    "shots": self.shots_count,
                },
            }
        )

        # 2. Evaluate landing position and emit point_suggested event
        suggestion = self._evaluate_point_suggestion(self.last_court_xy, self.last_shot)
        self._pending_events.append(
            {
                "ts": ts,
                "kind": "point_suggested",
                "data": suggestion,
            }
        )

    def reset_to_idle(self) -> None:
        """Transition END -> IDLE."""
        self.state = RallyState.IDLE
        self.current_t_ms = 0
        self.shots_count = 0
        self.last_court_xy = None
        self.last_shot = None

    def _evaluate_point_suggestion(
        self,
        landing_court_xy: list[float] | None,
        last_shot: ShotInfo | None,
    ) -> dict[str, Any]:
        """Estimate whether shuttle landed in/out and which side is recommended the point.

        NEVER modifies score automatically.
        """
        if landing_court_xy is None:
            return {
                "landing_court_xy": None,
                "landing_xy_m_est": None,
                "suggested_side": None,
                "in_bounds": None,
            }

        lx, ly = float(landing_court_xy[0]), float(landing_court_xy[1])

        y_min, y_max = (
            (0.0, self.COURT_WIDTH_M)
            if self.mode == "doubles"
            else (self.SINGLES_Y_MIN_M, self.SINGLES_Y_MAX_M)
        )
        in_bounds = 0.0 <= lx <= self.COURT_LENGTH_M and y_min <= ly <= y_max

        suggested_side: int | None = None

        if in_bounds:
            # Landed IN on opponent's court: opponent missed it -> other player wins point
            if lx > self.COURT_NET_X_M:
                # Landed on Side 1 court -> Side 0 wins point
                suggested_side = 0
            else:
                # Landed on Side 0 court -> Side 1 wins point
                suggested_side = 1
        else:
            # Landed OUT: who hit it out?
            hitter_side = (
                self._player_side(last_shot.by)
                if last_shot is not None and last_shot.by is not None
                else None
            )
            if hitter_side is not None:
                # The hitter hit the shuttle out -> opponent wins point
                suggested_side = 1 - hitter_side
            else:
                # Infer from direction: if past Side 1 backline, Side 0 hit it out
                if lx > self.COURT_LENGTH_M:
                    suggested_side = 1
                elif lx < 0.0:
                    suggested_side = 0
                else:
                    # Side out: if on Side 1 half, Side 0 likely hit it
                    suggested_side = 1 if lx > self.COURT_NET_X_M else 0

        return {
            "landing_court_xy": [round(lx, 2), round(ly, 2)],
            "landing_xy_m_est": [round(lx, 2), round(ly, 2)],
            "suggested_side": suggested_side,
            "in_bounds": in_bounds,
        }

    def _player_side(self, player_id: int) -> int | None:
        if self.mode == "singles":
            return player_id if player_id in (0, 1) else None
        return self.player_sides.get(player_id)
