"""Rally finite-state machine (IDLE -> SERVE -> RALLY -> END) and point recommendation.

According to CONTRACTS and T08:
- Computes rally active state, rally index (n), elapsed duration (t_ms), and shots count.
- At rally end, estimates landing point and emits 'point_suggested' event.
- STRICT RULE: Never modifies match score automatically; requires operator confirmation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from hud.fusion.homography import transform_point
from hud.fusion.stats import ShotInfo, drag_speed_kmh_est


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
        homography: list[list[float]] | None = None,
        end_gap_ms: float = 1800.0,
        landing_margin_m: float = 1.0,
        rest_landing_ms: float = 3000.0,
    ) -> None:
        if mode not in {"singles", "doubles"}:
            raise ValueError("mode must be 'singles' or 'doubles'")
        self.mode = mode
        # Image -> court homography; the caller keeps it in sync with calibration.
        self.homography = homography
        # A rally ends only after the shuttle stays unseen this long (LESSONS 6: 1.8 s).
        self.end_gap_ns = int(end_gap_ms * 1_000_000)
        # Landings further than this outside the court are homography/detection errors.
        self.landing_margin_m = landing_margin_m
        # A shuttle seen at rest this soon after the rally ended lies where it landed.
        # Proposed config key: `fusion.rally.rest_landing_ms`.
        self.rest_landing_ns = int(rest_landing_ms * 1_000_000)
        self._rest_landing_used = False
        self.state: RallyState = RallyState.IDLE
        self.rally_n: int = 0
        self.shots_count: int = 0
        self.start_ts: int = 0
        self.end_ts: int = 0
        self.current_t_ms: int = 0

        self.last_shuttle_status: str = "lost"
        self.last_seen_px: list[float] | None = None
        self.last_seen_ts: int = 0
        self.last_shot: ShotInfo | None = None
        # The shot last_shot answered; its speed_kmh_est is known now that it was returned.
        self.prev_shot: ShotInfo | None = None
        # Contract: set only once the rally has ended, null otherwise.
        self.landing_xy_m_est: list[float] | None = None

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
        """Transition IDLE/END -> SERVE."""
        if self.state in (RallyState.IDLE, RallyState.END):
            self.state = RallyState.SERVE
            self.rally_n += 1
            self.shots_count = 0
            self.start_ts = ts
            self.end_ts = 0
            self.current_t_ms = 0
            self.last_seen_px = None
            self.last_seen_ts = ts
            self.last_shot = None
            self.prev_shot = None
            self.landing_xy_m_est = None
            self._rest_landing_used = False
            self._pending_events.append(
                {"ts": ts, "kind": "rally_start", "data": {"n": self.rally_n}}
            )

    def start_rally(self, ts: int) -> None:
        """Transition SERVE/IDLE -> RALLY."""
        if self.state in (RallyState.IDLE, RallyState.END):
            self.start_serve(ts)
        self.state = RallyState.RALLY

    def on_shot(self, shot: ShotInfo) -> None:
        """Record a confirmed shot; shots must alternate court sides.

        Two consecutive shots from the same side mean one is spurious: keep the one
        with stronger evidence (audio-matched beats vision-only, then audio strength).
        Only past shots are compared, so this stays realtime.

        A shot's speed needs its return (LESSONS 7), so each accepted shot fills in
        ``speed_kmh_est`` of the shot it answers and reports it as ``prev_speed_kmh_est``.
        """
        if not self.is_active:
            self.start_serve(shot.ts)
        self.state = RallyState.RALLY

        prev = self.last_shot
        replaces: int | None = None
        if prev is not None and shot.side is not None and shot.side == prev.side:
            if shot.evidence <= prev.evidence:
                return
            replaces = prev.ts
        else:
            self.shots_count += 1
            self.prev_shot = prev
        self.last_shot = shot

        answered = self.prev_shot
        if answered is not None:
            # Recomputed on replacement, so a spurious return never keeps its speed.
            answered.speed_kmh_est = drag_speed_kmh_est(
                answered.hitter_xy_m, shot.hitter_xy_m, shot.ts - answered.ts
            )

        data: dict[str, Any] = {
            "by": shot.by,
            "speed_kmh_est": shot.speed_kmh_est,
            "shots": self.shots_count,
        }
        if replaces is not None:
            data["replaces_ts"] = replaces
        if answered is not None:
            data["prev_ts"] = answered.ts
            data["prev_speed_kmh_est"] = answered.speed_kmh_est
        self._pending_events.append({"ts": shot.ts, "kind": "shot", "data": data})

    def on_shuttle_update(
        self,
        status: str,
        img_xy: list[float] | None,
        ts: int,
        rest_px: list[float] | None = None,
    ) -> None:
        """Feed the Kalman status, the detected image point and the rest point every frame."""
        self.last_shuttle_status = status
        if status == "tracked" and img_xy is not None:
            self.last_seen_px = list(img_xy)
            self.last_seen_ts = ts
        if rest_px is not None and not self._rest_landing_used:
            self._on_rest(rest_px, ts)

        if not self.is_active:
            return
        if ts >= self.start_ts:
            self.current_t_ms = (ts - self.start_ts) // 1_000_000
        # A brief occlusion keeps the rally alive; a long loss after a shot ends it.
        if status == "lost" and self.shots_count >= 1 and ts - self.last_seen_ts >= self.end_gap_ns:
            self.end_rally(ts)

    def end_rally(self, ts: int) -> None:
        """Transition RALLY/SERVE -> END and suggest (never award) the point."""
        if not self.is_active:
            return

        self.state = RallyState.END
        self.end_ts = ts
        # Duration runs to the last sighting, not to the moment the loss was confirmed.
        last_ts = max(self.start_ts, min(ts, self.last_seen_ts))
        self.current_t_ms = (last_ts - self.start_ts) // 1_000_000
        self._emit_rally_end(ts)
        # A rally with no shot or no time after it is a false start: no point to suggest.
        if self.shots_count == 0 or self.current_t_ms == 0:
            return

        suggestion = self._evaluate_point_suggestion(self._landing_court_xy(), self.last_shot)
        self.landing_xy_m_est = suggestion["landing_xy_m_est"]
        self._pending_events.append({"ts": ts, "kind": "point_suggested", "data": suggestion})

    def _on_rest(self, rest_px: list[float], ts: int) -> None:
        # The rest point is the true landing; it beats the last in-flight sighting.
        if self.is_active:
            self.last_seen_px = list(rest_px)
            self._rest_landing_used = True
            return
        if self.state != RallyState.END or ts - self.end_ts > self.rest_landing_ns:
            return
        self._rest_landing_used = True
        if self.shots_count == 0 or self.current_t_ms == 0:
            return  # false start: no suggestion to correct
        self.last_seen_px = list(rest_px)
        landing = self._landing_court_xy()
        if landing is None:
            return
        suggestion = self._evaluate_point_suggestion(landing, self.last_shot)
        self.landing_xy_m_est = suggestion["landing_xy_m_est"]
        # Same ts as the original suggestion, so the operator's pending entry is replaced.
        self._pending_events.append(
            {"ts": self.end_ts, "kind": "point_suggested", "data": suggestion}
        )

    def _emit_rally_end(self, ts: int) -> None:
        # Every emitted rally_start gets exactly one rally_end, false starts included.
        self._pending_events.append(
            {
                "ts": ts,
                "kind": "rally_end",
                "data": {"n": self.rally_n, "t_ms": self.current_t_ms, "shots": self.shots_count},
            }
        )

    def reset_to_idle(self) -> None:
        """Transition any state -> IDLE; an active rally is closed with rally_end first."""
        if self.is_active:
            self._emit_rally_end(self.start_ts + self.current_t_ms * 1_000_000)
        self.state = RallyState.IDLE
        self.current_t_ms = 0
        self.shots_count = 0
        self.last_seen_px = None
        self.last_shot = None
        self.prev_shot = None
        self.landing_xy_m_est = None

    def _landing_court_xy(self) -> list[float] | None:
        # The last image point before the shuttle vanished is taken as on the floor.
        if self.homography is None or self.last_seen_px is None:
            return None
        try:
            lx, ly = transform_point(self.homography, self.last_seen_px)
        except ValueError:
            return None
        m = self.landing_margin_m
        if not (-m <= lx <= self.COURT_LENGTH_M + m and -m <= ly <= self.COURT_WIDTH_M + m):
            return None
        return [lx, ly]

    def _evaluate_point_suggestion(
        self,
        landing_court_xy: list[float] | None,
        last_shot: ShotInfo | None,
    ) -> dict[str, Any]:
        """Estimate in/out and which side (and team, if known) should get the point.

        NEVER modifies score automatically.
        """
        if landing_court_xy is None:
            return {
                "landing_xy_m_est": None,
                "suggested_side": None,
                "suggested_team": None,
                "in_bounds": None,
            }

        lx, ly = float(landing_court_xy[0]), float(landing_court_xy[1])

        y_min, y_max = (
            (0.0, self.COURT_WIDTH_M)
            if self.mode == "doubles"
            else (self.SINGLES_Y_MIN_M, self.SINGLES_Y_MAX_M)
        )
        in_bounds = 0.0 <= lx <= self.COURT_LENGTH_M and y_min <= ly <= y_max
        hitter_side = last_shot.side if last_shot is not None else None

        if in_bounds:
            # Landed in: the side it landed on loses the point.
            suggested_side = 0 if lx > self.COURT_NET_X_M else 1
        elif hitter_side is not None:
            # Hit out: the hitter's opponents win.
            suggested_side = 1 - hitter_side
        elif lx > self.COURT_LENGTH_M:
            suggested_side = 1
        elif lx < 0.0:
            suggested_side = 0
        else:
            # Side out with unknown hitter: assume it came from the opposite half.
            suggested_side = 1 if lx > self.COURT_NET_X_M else 0

        suggested_team: int | None = None
        if last_shot is not None and last_shot.side is not None and last_shot.team is not None:
            same = suggested_side == last_shot.side
            suggested_team = last_shot.team if same else 1 - last_shot.team

        return {
            "landing_xy_m_est": [round(lx, 2), round(ly, 2)],
            "suggested_side": suggested_side,
            "suggested_team": suggested_team,
            "in_bounds": in_bounds,
        }
