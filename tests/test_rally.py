from __future__ import annotations

import math

import pytest

from hud.fusion.kalman import ShuttleKalmanFilter
from hud.fusion.rally_fsm import RallyFSM, RallyState
from hud.fusion.stats import ShotDetector, ShotInfo, SpeedEstimator

# ============================================================================
# 1. Kalman Filter Tests
# ============================================================================


def test_kalman_tracked_state_updates_smoothly() -> None:
    kf = ShuttleKalmanFilter(max_predicted_frames=10)
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
    kf = ShuttleKalmanFilter(max_predicted_frames=10)
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
# 2. Shot Detector and Speed Estimator Tests
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


def test_speed_estimator_calculates_speed_over_3_to_5_frames() -> None:
    estimator = SpeedEstimator(min_frames=3, max_frames=5)
    ts = 1_000_000_000
    dt_ns = 20_000_000  # 50 fps -> 20ms per frame

    # Moving 1.0 meter per 20ms -> 50 m/s = 180 km/h
    estimator.start_sampling(ts, [2.0, 2.0])

    s2 = estimator.add_sample(ts + 1 * dt_ns, [3.0, 2.0])
    assert s2 is None  # 2 frames gathered (needs at least 3)

    s3 = estimator.add_sample(ts + 2 * dt_ns, [4.0, 2.0])
    assert s3 is None  # max_frames is 5

    s4 = estimator.add_sample(ts + 3 * dt_ns, [5.0, 2.0])
    assert s4 is None

    # 5th frame triggers speed calculation
    speed = estimator.add_sample(ts + 4 * dt_ns, [6.0, 2.0])
    assert speed is not None
    # 4 meters in 0.08s = 50 m/s = 180 km/h
    assert math.isclose(speed, 180.0, abs_tol=1.0)


def test_player_attribution_selects_closest_player() -> None:
    detector = ShotDetector()
    players = [
        {"id": 0, "court_xy_m": [2.0, 3.0]},
        {"id": 1, "court_xy_m": [11.0, 3.0]},
    ]
    # Shuttle at [2.5, 3.1] -> closest to Player 0
    attributed_0 = detector.attribute_player([2.5, 3.1], players)
    assert attributed_0 == 0

    # Shuttle at [10.5, 2.8] -> closest to Player 1
    attributed_1 = detector.attribute_player([10.5, 2.8], players)
    assert attributed_1 == 1


def test_player_attribution_supports_four_doubles_players() -> None:
    detector = ShotDetector()
    players = [
        {"id": 0, "court_xy_m": [2.0, 1.0]},
        {"id": 1, "court_xy_m": [2.0, 5.0]},
        {"id": 2, "court_xy_m": [11.0, 1.0]},
        {"id": 3, "court_xy_m": [10.5, 5.8]},
    ]

    assert detector.attribute_player([10.4, 5.7], players) == 3


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

    # Mid-rally temporary occlusion: 5 frames predicted -> rally stays active
    for _ in range(5):
        t += 16_666_667
        fsm.on_shuttle_update("predicted", [6.0, 2.5], t)
    assert fsm.is_active

    # Shuttle lost for > 10 frames -> transitions to lost -> rally ends
    t += 16_666_667
    fsm.on_shuttle_update("lost", None, t)
    assert fsm.state == RallyState.END
    assert not fsm.is_active

    events = fsm.drain_events()
    assert any(e["kind"] == "rally_end" for e in events)
    assert any(e["kind"] == "point_suggested" for e in events)


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
        [14.2, 3.0], last_shot=ShotInfo(ts=0, by=0, speed_kmh_est=None)
    )
    assert sugg_out["in_bounds"] is False
    assert sugg_out["suggested_side"] == 1

    # Case 4: Landed OUT across singles sideline ([5.0, 0.20]), hit by Player 1 -> Side 0 wins point
    sugg_side_out = fsm._evaluate_point_suggestion(
        [5.0, 0.20], last_shot=ShotInfo(ts=0, by=1, speed_kmh_est=None)
    )
    assert sugg_side_out["in_bounds"] is False
    assert sugg_side_out["suggested_side"] == 0
    assert sugg_side_out["landing_xy_m_est"] == [5.0, 0.2]


def test_doubles_point_suggestions_use_full_court_width_and_player_sides() -> None:
    fsm = RallyFSM(mode="doubles", player_sides={0: 0, 1: 0, 2: 1, 3: 1})

    landed_in_doubles_alley = fsm._evaluate_point_suggestion(
        [5.0, 0.20], last_shot=ShotInfo(ts=0, by=2, speed_kmh_est=None)
    )
    assert landed_in_doubles_alley["in_bounds"] is True
    assert landed_in_doubles_alley["suggested_side"] == 1

    landed_outside_doubles_sideline = fsm._evaluate_point_suggestion(
        [5.0, -0.01], last_shot=ShotInfo(ts=0, by=2, speed_kmh_est=None)
    )
    assert landed_outside_doubles_sideline["in_bounds"] is False
    assert landed_outside_doubles_sideline["suggested_side"] == 0


def test_rally_fsm_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must"):
        RallyFSM(mode="mixed")
