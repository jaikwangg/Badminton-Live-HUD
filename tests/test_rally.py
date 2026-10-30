from __future__ import annotations

import math
from typing import Any

import pytest

from hud.fusion.kalman import ShuttleKalmanFilter
from hud.fusion.rally_fsm import RallyFSM, RallyState
from hud.fusion.stats import ShotDetector, ShotInfo, drag_speed_kmh_est

# ============================================================================
# 1. Kalman Filter Tests
# ============================================================================


def test_kalman_tracked_state_updates_smoothly() -> None:
    kf = ShuttleKalmanFilter()
    ts_start = 1_000_000_000
    dt_ns = 16_666_667  # ~60 fps

    # Feed linear movement: (x, y) moving rightwards at 1000 px/s
    res = None
    for i in range(10):
        ts = ts_start + i * dt_ns
        px = [100.0 + i * 16.66, 200.0]
        res = kf.update(visible=True, px=px, ts=ts)
        assert res.status == "tracked"
        assert res.px is not None

    assert res is not None
    # X velocity should be approximately 1000 px/s
    assert abs(res.velocity[0] - 1000.0) < 150.0


def test_kalman_transitions_to_predicted_for_up_to_10_frames_then_lost() -> None:
    kf = ShuttleKalmanFilter()
    ts = 1_000_000_000
    dt_ns = 16_666_667

    # Initialize with 5 frames moving at ~1000 px/s in x
    for i in range(5):
        ts += dt_ns
        kf.update(visible=True, px=[100.0 + i * 16.0, 200.0], ts=ts)

    # Now shuttle disappears for 10 consecutive frames: must be 'predicted'
    last_x = 100.0 + 4 * 16.0
    for frame_idx in range(1, 11):
        ts += dt_ns
        res = kf.update(visible=False, px=None, ts=ts)
        assert res.status == "predicted", f"Frame {frame_idx} should be predicted"
        assert res.px is not None, f"Frame {frame_idx} should provide predicted coordinates"
        # Coordinates should continue forward along trajectory
        assert res.px[0] > last_x
        last_x = res.px[0]

    # 11th frame missing: must transition to 'lost' with px=None
    ts += dt_ns
    res_11 = kf.update(visible=False, px=None, ts=ts)
    assert res_11.status == "lost"
    assert res_11.px is None

    # Re-acquisition: when visible again, transitions back to 'tracked'
    ts += dt_ns
    res_reacquired = kf.update(visible=True, px=[300.0, 200.0], ts=ts)
    assert res_reacquired.status == "tracked"
    assert res_reacquired.px == [300.0, 200.0]


def test_kalman_detects_direction_change_on_reversal() -> None:
    kf = ShuttleKalmanFilter(direction_change_deg=50.0)
    ts = 1_000_000_000
    dt_ns = 16_666_667

    # Shuttle flying rightwards at 1200 px/s
    for i in range(8):
        ts += dt_ns
        kf.update(visible=True, px=[100.0 + i * 20.0, 200.0], ts=ts)

    # Racket hit reverses direction leftwards at 1200 px/s
    dir_changed = False
    for i in range(5):
        ts += dt_ns
        res = kf.update(visible=True, px=[240.0 - (i + 1) * 20.0, 200.0], ts=ts)
        if res.direction_changed:
            dir_changed = True
            break

    assert dir_changed, "Filter should detect direction reversal"


# ============================================================================
# 2. Shot Detector and Shot Speed Tests
# ============================================================================


def test_shot_detector_confirms_when_audio_and_vision_within_100ms() -> None:
    detector = ShotDetector(match_window_ms=100.0)
    t_base = 5_000_000_000

    # Vision direction change at t_base
    shot_pre = detector.register_direction_change(ts=t_base, court_xy=[4.0, 3.0])
    assert shot_pre is None  # Waiting for matching audio

    # Audio hit arrives 40ms later (< 100ms)
    shot = detector.register_audio_hit(ts=t_base + 40_000_000, strength=0.8)
    assert shot is not None
    assert shot.ts == t_base
    assert shot.court_xy_m == [4.0, 3.0]


def test_shot_detector_rejects_when_audio_and_vision_differ_by_more_than_100ms() -> None:
    detector = ShotDetector(match_window_ms=100.0)
    t_base = 5_000_000_000

    # Vision direction change at t_base
    detector.register_direction_change(ts=t_base, court_xy=[4.0, 3.0])

    # Audio hit arrives 150ms later (> 100ms match window)
    shot = detector.register_audio_hit(ts=t_base + 150_000_000, strength=0.8)
    assert shot is None


def test_shot_detector_degraded_mode_counts_shot_from_vision_alone() -> None:
    detector = ShotDetector()
    t_base = 5_000_000_000

    # In degraded audio mode, direction change immediately confirms shot
    shot = detector.register_direction_change(
        ts=t_base,
        court_xy=[6.0, 2.5],
        audio_degraded=True,
    )
    assert shot is not None
    assert shot.ts == t_base
    assert shot.court_xy_m == [6.0, 2.5]


def test_drag_speed_matches_hand_computed_values() -> None:
    # k = 9.81 / 6.8^2 = 0.212154; v0 = (e^(kD) - 1) / (kT)
    # D=10 m, T=1 s: (e^2.12154 - 1) / 0.212154 = 7.34397 / 0.212154 = 34.616 m/s = 124.6 km/h
    assert drag_speed_kmh_est([1.0, 3.0], [11.0, 3.0], 1_000_000_000) == pytest.approx(124.6)
    # D=6 m (3-4-5 x2), T=0.5 s: (e^1.272924 - 1) / 0.106077 = 24.240 m/s = 87.3 km/h
    assert drag_speed_kmh_est([0.0, 0.0], [3.6, 4.8], 500_000_000) == pytest.approx(87.3)


def test_drag_speed_is_null_when_unknown_or_implausible() -> None:
    assert drag_speed_kmh_est(None, [11.0, 3.0], 1_000_000_000) is None
    assert drag_speed_kmh_est([1.0, 3.0], None, 1_000_000_000) is None
    assert drag_speed_kmh_est([1.0, 3.0], [11.0, 3.0], 0) is None
    assert drag_speed_kmh_est([1.0, 3.0], [11.0, 3.0], -1) is None
    assert drag_speed_kmh_est([1.0, 3.0], [11.0, 3.0], 300_000_000) is None  # too short
    assert drag_speed_kmh_est([1.0, 3.0], [11.0, 3.0], 3_500_000_000) is None  # too long
    assert drag_speed_kmh_est([6.0, 3.0], [8.0, 3.0], 1_000_000_000) is None  # net play


def test_shot_speed_is_filled_for_previous_shot_when_it_is_returned() -> None:
    detector = ShotDetector()
    fsm = RallyFSM()

    def hit(ts: int, pid: int, side: int, xy: list[float]) -> ShotInfo:
        players = [{"id": pid, "side": side, "img_xy": [0, 0], "court_xy_m": xy}]
        shot = detector.register_direction_change(
            ts, px=[0, 0], players=players, audio_degraded=True
        )
        assert shot is not None and shot.hitter_xy_m == xy
        fsm.on_shot(shot)
        return shot

    s1 = hit(1_000_000_000, 0, 0, [1.0, 3.0])
    _rally_start, event = fsm.drain_events()
    assert s1.speed_kmh_est is None and event["data"]["speed_kmh_est"] is None
    assert "prev_speed_kmh_est" not in event["data"]

    # The return fills in shot 1; the new shot's own speed stays unknown.
    s2 = hit(2_000_000_000, 1, 1, [11.0, 3.0])
    (event,) = fsm.drain_events()
    assert s1.speed_kmh_est == pytest.approx(124.6)
    assert event["data"]["prev_ts"] == s1.ts
    assert event["data"]["prev_speed_kmh_est"] == pytest.approx(124.6)
    assert s2.speed_kmh_est is None and fsm.prev_shot is s1 and fsm.last_shot is s2

    # A stronger same-side shot replaces s2 and recomputes s1 from the new return.
    s2b = ShotInfo(ts=2_500_000_000, by=1, speed_kmh_est=None, side=1, evidence=1.5)
    s2b.hitter_xy_m = [7.0, 3.0]  # D=6 m in 1.5 s
    fsm.on_shot(s2b)
    (event,) = fsm.drain_events()
    assert event["data"]["replaces_ts"] == s2.ts and event["data"]["prev_ts"] == s1.ts
    expected = drag_speed_kmh_est([1.0, 3.0], [7.0, 3.0], 1_500_000_000)
    assert s1.speed_kmh_est == expected == event["data"]["prev_speed_kmh_est"]

    # Unknown returner position -> null, never a guessed number.
    s3 = ShotInfo(ts=3_500_000_000, by=0, speed_kmh_est=None, side=0)
    fsm.on_shot(s3)
    assert s2b.speed_kmh_est is None and fsm.drain_events()[0]["data"]["prev_speed_kmh_est"] is None


def test_player_attribution_selects_player_nearest_shuttle_in_image() -> None:
    detector = ShotDetector()
    players = [
        {"id": 0, "side": 0, "bbox": [100, 500, 200, 700], "img_xy": [150, 700]},
        {"id": 1, "side": 1, "bbox": [600, 100, 650, 220], "img_xy": [625, 220]},
    ]
    # Inside player 0's bbox -> distance 0.
    assert detector.attribute_player([150, 520], players)["id"] == 0
    assert detector.attribute_player([640, 90], players)["id"] == 1
    # No bbox: fall back to the foot point.
    assert detector.attribute_player([600, 230], [{"id": 1, "img_xy": [625, 220]}])["id"] == 1
    assert detector.attribute_player(None, players) is None


def test_doubles_hitter_is_nearest_player_on_that_side_at_hit_time() -> None:
    detector = ShotDetector()
    players = [
        {"id": 0, "team": 0, "side": 0, "bbox": [100, 500, 200, 700]},
        {"id": 1, "team": 0, "side": 0, "bbox": [800, 500, 900, 700]},
        {"id": 2, "team": 1, "side": 1, "bbox": [300, 100, 340, 200]},
        {"id": 3, "team": 1, "side": 1, "bbox": [600, 100, 640, 200]},
    ]
    shot = detector.register_direction_change(
        ts=1_000_000_000, px=[650, 90], players=players, audio_degraded=True
    )
    assert shot is not None
    assert (shot.by, shot.side, shot.team, shot.evidence) == (3, 1, 1, 0.0)

    # Audio-matched shot carries stronger evidence and the hitter of the stored change.
    detector.register_direction_change(ts=2_000_000_000, px=[880, 520], players=players)
    shot = detector.register_audio_hit(ts=2_030_000_000, strength=0.6)
    assert shot is not None
    assert (shot.by, shot.side) == (1, 0)
    assert math.isclose(shot.evidence, 1.6)


# ============================================================================
# 3. Rally State Machine Tests
# ============================================================================


def test_rally_fsm_lifecycle_transitions() -> None:
    fsm = RallyFSM()
    assert fsm.state == RallyState.IDLE
    assert not fsm.is_active

    # 1. Start serve at t = 10s
    t_start = 10_000_000_000
    fsm.start_serve(t_start)
    assert fsm.state == RallyState.SERVE
    assert fsm.is_active
    assert fsm.rally_n == 1
    assert fsm.shots_count == 0

    events = fsm.drain_events()
    assert len(events) == 1
    assert events[0]["kind"] == "rally_start"

    # 2. First shot: server hits the shuttle -> RALLY
    t_shot1 = t_start + 500_000_000
    fsm.on_shot(ShotInfo(ts=t_shot1, by=0, speed_kmh_est=160.0, court_xy_m=[3.0, 2.5]))
    assert fsm.state == RallyState.RALLY
    assert fsm.shots_count == 1

    # 3. Return shot by Player 1
    t_shot2 = t_shot1 + 800_000_000
    fsm.on_shot(ShotInfo(ts=t_shot2, by=1, speed_kmh_est=210.0, court_xy_m=[9.5, 3.0]))
    assert fsm.shots_count == 2

    # 4. Update during rally updates t_ms
    t_mid = t_shot2 + 200_000_000
    fsm.on_shuttle_update("tracked", [7.0, 3.0], t_mid)
    rally_dict = fsm.get_rally_dict()
    assert rally_dict["active"] is True
    assert rally_dict["shots"] == 2
    assert rally_dict["t_ms"] == 1500  # 1.5 seconds

    # 5. End rally at t = 12s
    t_end = t_start + 2_000_000_000
    fsm.end_rally(t_end)
    assert fsm.state == RallyState.END
    assert not fsm.is_active

    end_events = fsm.drain_events()
    kinds = [e["kind"] for e in end_events]
    assert "rally_end" in kinds
    assert "point_suggested" in kinds


def test_rally_fsm_shuttle_lost_during_rally_concludes_rally() -> None:
    fsm = RallyFSM()
    t = 1_000_000_000
    fsm.start_serve(t)

    # 2 shots occurred
    t += 500_000_000
    fsm.on_shot(ShotInfo(ts=t, by=0, speed_kmh_est=150.0, court_xy_m=[3.0, 3.0]))
    t += 700_000_000
    fsm.on_shot(ShotInfo(ts=t, by=1, speed_kmh_est=180.0, court_xy_m=[8.0, 2.0]))

    fsm.on_shuttle_update("tracked", [500.0, 300.0], t)
    last_seen = t

    # Mid-rally occlusion: predicted, then lost for under the end gap -> rally stays active
    for _ in range(5):
        t += 16_666_667
        fsm.on_shuttle_update("predicted", [510.0, 310.0], t)
    t += 1_000_000_000
    fsm.on_shuttle_update("lost", None, t)
    assert fsm.is_active

    # Shuttle reappears: still the same rally
    t += 16_666_667
    fsm.on_shuttle_update("tracked", [520.0, 320.0], t)
    last_seen = t
    assert fsm.state == RallyState.RALLY

    # Lost for the full end gap -> rally ends
    t += 1_800_000_000
    fsm.on_shuttle_update("lost", None, t)
    assert fsm.state == RallyState.END
    assert not fsm.is_active
    assert fsm.current_t_ms == (last_seen - 1_000_000_000) // 1_000_000

    events = fsm.drain_events()
    assert any(e["kind"] == "rally_end" for e in events)
    assert any(e["kind"] == "point_suggested" for e in events)


def test_rally_fsm_end_returns_to_idle_and_next_serve_starts_new_rally() -> None:
    fsm = RallyFSM()
    ms = 1_000_000
    fsm.on_shot(ShotInfo(ts=1_000 * ms, by=0, speed_kmh_est=None))  # IDLE -> RALLY directly
    assert fsm.state == RallyState.RALLY
    fsm.on_shuttle_update("tracked", [1.0, 1.0], 1_500 * ms)
    fsm.end_rally(2_000 * ms)
    fsm.end_rally(3_000 * ms)  # ignored when not active
    assert sum(e["kind"] == "rally_end" for e in fsm.drain_events()) == 1
    fsm.reset_to_idle()
    assert fsm.state == RallyState.IDLE
    fsm.start_serve(4_000 * ms)
    assert (fsm.state, fsm.rally_n, fsm.shots_count) == (RallyState.SERVE, 2, 0)
    fsm.end_rally(5_000 * ms)  # SERVE -> END
    fsm.start_rally(6_000 * ms)  # END -> SERVE -> RALLY
    assert (fsm.state, fsm.rally_n) == (RallyState.RALLY, 3)


def test_rally_fsm_shots_alternate_sides_keeping_stronger_evidence() -> None:
    fsm = RallyFSM()
    fsm.on_shot(ShotInfo(ts=1, by=0, speed_kmh_est=None, side=0, evidence=1.5))
    fsm.on_shot(ShotInfo(ts=2, by=1, speed_kmh_est=None, side=1, evidence=0.0))
    assert fsm.shots_count == 2
    fsm.drain_events()

    # Same side again with weaker evidence -> dropped.
    fsm.on_shot(ShotInfo(ts=3, by=1, speed_kmh_est=None, side=1, evidence=0.0))
    assert fsm.shots_count == 2
    assert fsm.last_shot is not None and fsm.last_shot.ts == 2
    assert fsm.drain_events() == []

    # Same side with stronger evidence -> replaces the previous shot, count unchanged.
    fsm.on_shot(ShotInfo(ts=4, by=1, speed_kmh_est=None, side=1, evidence=1.2))
    assert fsm.shots_count == 2
    assert fsm.last_shot is not None and fsm.last_shot.ts == 4
    (event,) = fsm.drain_events()
    assert event["data"]["replaces_ts"] == 2

    fsm.on_shot(ShotInfo(ts=5, by=0, speed_kmh_est=None, side=0))
    assert fsm.shots_count == 3


def test_rally_end_estimates_landing_from_last_seen_point_via_homography() -> None:
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    fsm = RallyFSM(mode="doubles", homography=identity)
    ms = 1_000_000
    fsm.on_shot(ShotInfo(ts=1_000 * ms, by=1, speed_kmh_est=None, side=0, team=0))
    fsm.on_shuttle_update("tracked", [9.0, 3.0], 2_000 * ms)
    assert fsm.landing_xy_m_est is None  # null while the shuttle is in flight
    fsm.on_shuttle_update("predicted", [9.5, 3.0], 3_000 * ms)  # predictions are not landings
    fsm.end_rally(4_000 * ms)

    assert fsm.landing_xy_m_est == [9.0, 3.0]
    suggestion = fsm.drain_events()[-1]["data"]
    assert suggestion["suggested_side"] == 0
    assert suggestion["suggested_team"] == 0

    fsm.start_serve(5_000 * ms)
    assert fsm.landing_xy_m_est is None


def test_rally_end_drops_landing_far_outside_court() -> None:
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ms = 1_000_000
    for point, expected in (
        ([15.42, 2.78], None),
        ([20.36, 2.09], None),
        ([14.0, 3.0], [14.0, 3.0]),
    ):
        fsm = RallyFSM(homography=identity)
        fsm.on_shot(ShotInfo(ts=1_000 * ms, by=0, speed_kmh_est=None, side=0))
        fsm.on_shuttle_update("tracked", point, 2_000 * ms)
        fsm.end_rally(3_000 * ms)
        assert fsm.landing_xy_m_est == expected
    fsm.reset_to_idle()
    assert fsm.landing_xy_m_est is None


def test_false_start_rally_is_closed_without_point_suggestion() -> None:
    ms = 1_000_000
    fsm = RallyFSM()
    fsm.start_serve(1_000 * ms)
    fsm.drain_events()
    fsm.end_rally(2_000 * ms)  # no shot
    assert fsm.state == RallyState.END
    assert [e["kind"] for e in fsm.drain_events()] == ["rally_end"]

    # A shot but the shuttle never seen after it: t_ms is 0, still paired with rally_end.
    fsm.on_shot(ShotInfo(ts=3_000 * ms, by=0, speed_kmh_est=None))
    fsm.on_shuttle_update("lost", None, 5_000 * ms)
    assert fsm.state == RallyState.END
    events = fsm.drain_events()
    assert [e["kind"] for e in events] == ["rally_start", "shot", "rally_end"]
    assert events[-1]["data"] == {"n": 2, "t_ms": 0, "shots": 1}


def test_operator_reset_mid_rally_closes_it_with_rally_end() -> None:
    ms = 1_000_000
    fsm = RallyFSM()
    fsm.on_shot(ShotInfo(ts=1_000 * ms, by=0, speed_kmh_est=None))
    fsm.on_shuttle_update("tracked", [1.0, 1.0], 1_700 * ms)
    fsm.reset_to_idle()
    events = fsm.drain_events()
    assert [e["kind"] for e in events] == ["rally_start", "shot", "rally_end"]
    assert events[-1]["ts"] == 1_700 * ms
    fsm.reset_to_idle()  # already idle: nothing more
    assert fsm.drain_events() == []


def test_kalman_predicted_and_lost_follow_time_not_message_count() -> None:
    # Inference publishes every 50 ms: 3 misses (150 ms) are predicted, the 4th (200 ms) is lost.
    kf = ShuttleKalmanFilter()
    step = 50_000_000
    for i in range(5):
        res = kf.update(True, [100.0 + i * 50.0, 200.0], i * step)  # 1000 px/s
    assert abs(res.velocity[0] - 1000.0) < 150.0  # real dt, not 1/60 s
    statuses = [kf.update(False, None, (4 + k) * step).status for k in range(1, 5)]
    assert statuses == ["predicted", "predicted", "predicted", "lost"]

    kf = ShuttleKalmanFilter(predict_ms=300.0)
    kf.update(True, [100.0, 200.0], 0)
    assert kf.update(False, None, 250_000_000).status == "predicted"
    assert kf.update(False, None, 350_000_000).status == "lost"


def test_kalman_rejects_jumps_for_at_most_three_frames() -> None:
    kf = ShuttleKalmanFilter(max_px_per_s=3000.0)
    dt_ns = 16_666_667
    for i in range(5):
        kf.update(True, [100.0 + i * 10.0, 200.0], i * dt_ns)  # 600 px/s

    # A far-away false detection is replaced by the prediction for 3 frames.
    for i in range(5, 8):
        res = kf.update(True, [1500.0, 900.0], i * dt_ns)
        assert res.status == "predicted"
        assert res.px is not None and res.px[0] < 200.0
    assert kf.last_seen_px == [140.0, 200.0]

    # The 4th consecutive jump is accepted: a real new trajectory is not blocked forever.
    res = kf.update(True, [1500.0, 900.0], 8 * dt_ns)
    assert res.status == "tracked"
    assert kf.last_seen_px == [1500.0, 900.0]

    # A single outlier does not wipe the track.
    kf.update(True, [1510.0, 900.0], 9 * dt_ns)
    kf.update(True, [100.0, 100.0], 10 * dt_ns)
    res = kf.update(True, [1530.0, 900.0], 11 * dt_ns)
    assert res.status == "tracked"


def test_point_suggested_in_bounds_and_out_bounds() -> None:
    fsm = RallyFSM()

    # Case 1: Landed IN on Side 1 court ([9.0, 3.0]) -> Side 0 wins point
    sugg_in = fsm._evaluate_point_suggestion(
        [9.0, 3.0], last_shot=ShotInfo(ts=0, by=0, speed_kmh_est=None)
    )
    assert sugg_in["in_bounds"] is True
    assert sugg_in["suggested_side"] == 0

    # Case 2: Landed IN on Side 0 court ([3.0, 3.0]) -> Side 1 wins point
    sugg_in_0 = fsm._evaluate_point_suggestion(
        [3.0, 3.0], last_shot=ShotInfo(ts=0, by=1, speed_kmh_est=None)
    )
    assert sugg_in_0["in_bounds"] is True
    assert sugg_in_0["suggested_side"] == 1

    # Case 3: Landed OUT over Side 1 backline ([14.2, 3.0]), hit by Player 0 -> Side 1 wins point
    sugg_out = fsm._evaluate_point_suggestion(
        [14.2, 3.0], last_shot=ShotInfo(ts=0, by=0, speed_kmh_est=None, side=0)
    )
    assert sugg_out["in_bounds"] is False
    assert sugg_out["suggested_side"] == 1

    # Case 4: Landed OUT across singles sideline ([5.0, 0.20]), hit by Player 1 -> Side 0 wins point
    sugg_side_out = fsm._evaluate_point_suggestion(
        [5.0, 0.20], last_shot=ShotInfo(ts=0, by=1, speed_kmh_est=None, side=1)
    )
    assert sugg_side_out["in_bounds"] is False
    assert sugg_side_out["suggested_side"] == 0
    assert sugg_side_out["landing_xy_m_est"] == [5.0, 0.2]


def test_doubles_point_suggestions_use_full_court_width_and_hitter_side() -> None:
    fsm = RallyFSM(mode="doubles")
    # Player 2 (team 0) is on side 1 now: side, not id or team, decides out calls.
    hit = ShotInfo(ts=0, by=2, speed_kmh_est=None, side=1, team=0)

    landed_in_doubles_alley = fsm._evaluate_point_suggestion([5.0, 0.20], last_shot=hit)
    assert landed_in_doubles_alley["in_bounds"] is True
    assert landed_in_doubles_alley["suggested_side"] == 1
    assert landed_in_doubles_alley["suggested_team"] == 0

    landed_outside_doubles_sideline = fsm._evaluate_point_suggestion([5.0, -0.01], last_shot=hit)
    assert landed_outside_doubles_sideline["in_bounds"] is False
    assert landed_outside_doubles_sideline["suggested_side"] == 0
    assert landed_outside_doubles_sideline["suggested_team"] == 1


def test_rally_fsm_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must"):
        RallyFSM(mode="mixed")


# ============================================================================
# 4. Replay: recorded worker messages through kalman + detector + fsm
# ============================================================================

Message = tuple[str, int, Any]  # (kind, ts_ms, payload)


def replay(
    messages: list[Message],
    *,
    audio_ok: bool = False,
    kf: ShuttleKalmanFilter | None = None,
    homography: list[list[float]] | None = None,
) -> list[dict[str, Any]]:
    """Feed shuttle/hit/players messages the way fusion does and return the emitted events.

    Payloads: shuttle -> image [x, y] or None (not visible), hit -> strength,
    players -> list of tracked player dicts (id, side, team, bbox, court_xy_m).
    """
    kf = kf or ShuttleKalmanFilter()
    detector = ShotDetector()
    fsm = RallyFSM(homography=homography)
    players: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for kind, ts_ms, payload in messages:
        ts = ts_ms * 1_000_000
        if kind == "players":
            players = payload
        elif kind == "hit":
            shot = detector.register_audio_hit(ts, payload)
            if shot is not None:
                fsm.on_shot(shot)
        else:
            res = kf.update(payload is not None, payload, ts)
            fsm.on_shuttle_update(res.status, kf.last_seen_px, ts, kf.rest_px)
            if res.direction_changed:
                shot = detector.register_direction_change(
                    ts, px=res.px, players=players, audio_degraded=not audio_ok
                )
                if shot is not None:
                    fsm.on_shot(shot)
        events.extend(fsm.drain_events())
    return events


_REPLAY_PLAYERS = [
    {"id": 0, "side": 0, "team": 0, "bbox": [350.0, 550.0, 450.0, 700.0], "court_xy_m": [2.0, 3.0]},
    {
        "id": 1,
        "side": 1,
        "team": 1,
        "bbox": [350.0, 200.0, 450.0, 300.0],
        "court_xy_m": [11.0, 3.0],
    },
]
# Synthetic ~4 Hz shuttle stream (CPU-rate inference): near player (bottom of image) serves
# up, far player returns at 1250 ms, the near player's return at ~2000 ms is hidden by one
# missed message, then the shuttle leaves view. A lone false reversal follows at 10.5 s.
_REPLAY_SHUTTLE = [
    (0, [400.0, 560.0]),
    (250, [400.0, 480.0]),
    (500, [400.0, 400.0]),
    (750, [400.0, 320.0]),
    (1000, [400.0, 260.0]),
    (1250, [400.0, 340.0]),
    (1500, [400.0, 420.0]),
    (1750, [400.0, 500.0]),
    (2000, None),
    (2250, [400.0, 470.0]),
    (2500, [400.0, 390.0]),
    (2750, [400.0, 310.0]),
    (3000, [400.0, 250.0]),
    *((ms, None) for ms in range(3250, 10000, 250)),
    (10000, [400.0, 560.0]),
    (10250, [400.0, 480.0]),
    (10500, [400.0, 560.0]),
    *((ms, None) for ms in range(10750, 13000, 250)),
]
_REPLAY_HITS = [(1260, 0.8), (2200, 0.7), (10490, 0.5)]


def _replay_fixture(*, with_hits: bool) -> list[Message]:
    messages: list[Message] = [("players", 0, _REPLAY_PLAYERS)]
    messages += [("shuttle", ms, px) for ms, px in _REPLAY_SHUTTLE]
    if with_hits:
        messages += [("hit", ms, strength) for ms, strength in _REPLAY_HITS]
    return sorted(messages, key=lambda m: m[1])


@pytest.mark.parametrize("audio_ok", [False, True])
def test_replay_low_rate_stream_counts_shots_and_pairs_rally_events(audio_ok: bool) -> None:
    events = replay(_replay_fixture(with_hits=audio_ok), audio_ok=audio_ok)
    assert [e["kind"] for e in events] == [
        "rally_start", "shot", "shot", "rally_end", "point_suggested",
        "rally_start", "shot", "rally_end",
    ]  # fmt: skip
    shots = [e for e in events if e["kind"] == "shot"]
    assert [s["data"]["by"] for s in shots] == [1, 0, 0]
    # The hit hidden by the missed message is reported on reappearance.
    assert [s["ts"] // 1_000_000 for s in shots[:2]] == [1250, 2200 if audio_ok else 2250]
    assert events[3]["data"] == {"n": 1, "t_ms": 1750, "shots": 2}
    assert events[-1]["data"] == {"n": 2, "t_ms": 0, "shots": 1}


def test_replay_without_direction_memory_misses_the_hidden_hit() -> None:
    events = replay(
        _replay_fixture(with_hits=False), kf=ShuttleKalmanFilter(direction_max_gap_ms=0.0)
    )
    first_rally_end = next(e for e in events if e["kind"] == "rally_end")
    assert first_rally_end["data"]["shots"] == 1


# Image px -> court m for the rest tests: (400, 516) -> (8.0, 5.16), inside the singles court.
_SCALE_H = [[0.02, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]]
# One shot, then the shuttle falls and lies at (400, 516) with +-5 px detector jitter; the
# jitter alone (56 px/s, reversing every message) would read as direction changes in flight.
_REST_SHUTTLE = [
    *((ms, px) for ms, px in _REPLAY_SHUTTLE if ms <= 1750),
    (2000, [400.0, 516.0]),
    *((ms, [405.0, 521.0] if ms % 500 else [395.0, 511.0]) for ms in range(2250, 8000, 250)),
]


def _rest_fixture(shuttle: list[tuple[int, list[float] | None]]) -> list[Message]:
    return [("players", 0, _REPLAY_PLAYERS)] + [("shuttle", ms, px) for ms, px in shuttle]


def test_replay_resting_shuttle_ends_rally_and_marks_landing() -> None:
    events = replay(_rest_fixture(_REST_SHUTTLE), homography=_SCALE_H)
    assert [e["kind"] for e in events] == ["rally_start", "shot", "rally_end", "point_suggested"]
    # Rest detected at 2500 ms; the rally runs from the 1250 ms shot to the 2250 ms sighting.
    assert events[2]["data"] == {"n": 1, "t_ms": 1000, "shots": 1}
    assert events[3]["data"]["landing_xy_m_est"] == [8.0, 5.16]


def test_replay_without_rest_detection_never_ends_rally() -> None:
    events = replay(
        _rest_fixture(_REST_SHUTTLE), kf=ShuttleKalmanFilter(rest_ms=1e9), homography=_SCALE_H
    )
    assert "rally_end" not in [e["kind"] for e in events]


@pytest.mark.parametrize(("rest_ms", "expected"), [(400.0, 1), (1e9, 22)])
def test_floor_jitter_gives_no_direction_changes_once_resting(
    rest_ms: float, expected: int
) -> None:
    kf = ShuttleKalmanFilter(rest_ms=rest_ms)
    changes = sum(
        kf.update(px is not None, px, ms * 1_000_000).direction_changed for ms, px in _REST_SHUTTLE
    )
    assert changes == expected  # 1 = the real hit at 1250 ms


def test_replay_rest_after_rally_end_replaces_landing_suggestion() -> None:
    # Last in-flight sighting at 3000 ms (far court), shuttle unseen until the rally ends
    # at 5000 ms, then seen lying at (400, 516) from 5250 ms.
    shuttle = [
        *((ms, px) for ms, px in _REPLAY_SHUTTLE if ms <= 3000),
        *((ms, None) for ms in range(3250, 5250, 250)),
        *((ms, [400.0, 516.0]) for ms in range(5250, 7000, 250)),
    ]
    events = replay(_rest_fixture(shuttle), homography=_SCALE_H)
    kinds = [e["kind"] for e in events]
    assert kinds == [
        "rally_start",
        "shot",
        "shot",
        "rally_end",
        "point_suggested",
        "point_suggested",
    ]
    first, second = events[4], events[5]
    assert first["data"]["landing_xy_m_est"] == [8.0, 2.5]
    assert second["ts"] == first["ts"]  # same key: replaces the pending suggestion
    assert second["data"]["landing_xy_m_est"] == [8.0, 5.16]


@pytest.mark.parametrize("rate_hz", [4, 60])
def test_kalman_slow_flying_shuttle_is_not_resting(rate_hz: int) -> None:
    kf = ShuttleKalmanFilter()
    step_ms = 1000 // rate_hz
    for i in range(3 * rate_hz):
        t_ms = i * step_ms
        res = kf.update(True, [100.0 + 0.04 * t_ms, 300.0], t_ms * 1_000_000)  # 40 px/s
        assert res.status == "tracked"
        assert kf.rest_px is None


def test_kalman_stationary_shuttle_becomes_lost_after_rest_ms() -> None:
    kf = ShuttleKalmanFilter(rest_radius_px=8.0, rest_ms=400.0)
    ms = 1_000_000
    statuses = [kf.update(True, [450.0, 516.0], t * 100 * ms).status for t in range(6)]
    assert statuses == ["tracked"] * 4 + ["lost"] * 2
    assert kf.rest_px == [450.0, 516.0]
    assert kf.update(False, None, 700 * ms).status == "lost"
    # Picked up and served: a detection outside the radius is in flight again.
    assert kf.update(True, [450.0, 480.0], 800 * ms).status == "tracked"
    assert kf.rest_px is None
