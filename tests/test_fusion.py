from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from hud.bus.messages import HeartbeatMessage, PlayersMessage, ShuttleMessage
from hud.fusion.homography import compute_homography, transform_point
from hud.fusion.state import FusionState, _min_cost_pairs, validate_game_state
from hud.server.operator_api import create_operator_app
from hud.server.ws_server import create_ws_app, ingest_worker_messages, offer_latest, run_at_rate


def test_homography_maps_reference_corners_within_one_centimeter() -> None:
    corners = [[100, 50], [900, 50], [900, 450], [100, 450]]
    transform = compute_homography(corners)
    point = transform_point(transform, [500, 250])
    assert point == pytest.approx([6.7, 3.05], abs=0.01)


def test_homography_maps_baselines_to_width_and_sidelines_to_length() -> None:
    # Clockwise from far-left: far-left, far-right, near-right, near-left.
    corners = [[300, 100], [700, 100], [900, 500], [100, 500]]
    transform = compute_homography(corners)
    expected = [[13.40, 0.0], [13.40, 6.10], [0.0, 6.10], [0.0, 0.0]]
    for corner, court_xy in zip(corners, expected, strict=True):
        assert transform_point(transform, corner) == pytest.approx(court_xy, abs=0.01)
    # The court center is where the image diagonals cross.
    net_center = transform_point(transform, [500, 700 / 3])
    assert net_center == pytest.approx([6.70, 3.05], abs=0.01)


def test_state_matches_contract_and_emits_null_unknowns(tmp_path) -> None:
    state = FusionState(tmp_path / "court_calib.json")
    state.ingest(
        PlayersMessage(
            frame_id=1,
            ts=1_000_000,
            players=[{"track_id": 10, "bbox": [10, 20, 30, 60], "foot_px": [20, 60], "conf": 0.9}],
        ).to_json()
    )
    state.ingest(HeartbeatMessage(worker="players", ts=1_000_000, fps=60.0, proc_ms=5.0).to_json())
    snapshot = state.snapshot()
    assert set(snapshot) == {"v", "ts", "match", "rally", "players", "shuttle", "events", "health"}
    assert snapshot["v"] == 1
    assert snapshot["players"][0]["court_xy_m"] is None
    assert snapshot["players"][0]["speed_mps"] is None
    assert snapshot["players"][1]["img_xy"] is None
    assert snapshot["health"]["workers"]["players"] == "ok"
    assert json.loads(json.dumps(snapshot)) == snapshot
    validate_game_state(snapshot)
    snapshot["players"].pop()
    with pytest.raises(ValueError):
        validate_game_state(snapshot)


def test_worker_without_recent_heartbeat_is_reported_down(tmp_path) -> None:
    state = FusionState(tmp_path / "court_calib.json")
    state.ingest(
        PlayersMessage(
            frame_id=1,
            ts=1_000_000,
            players=[],
        ).to_json()
    )
    assert state.snapshot()["health"]["workers"]["players"] == "down"


def test_calibrate_persists_matrix_and_score_swap_updates(tmp_path) -> None:
    calibration_path = tmp_path / "court_calib.json"
    state = FusionState(calibration_path)
    state.calibrate([[0, 0], [100, 0], [100, 50], [0, 50]])
    state.set_players(
        [
            {"id": 0, "name": "A", "color": "#fff", "side": 0},
            {"id": 1, "name": "B", "color": "#000", "side": 1},
        ]
    )
    state.add_score(0, 1)
    state.swap_ids(0, 1)
    snapshot = state.snapshot()
    assert snapshot["match"]["score"] == [1, None]
    assert calibration_path.exists()


def test_doubles_game_state_contains_four_players_and_two_per_side(tmp_path) -> None:
    state = FusionState(tmp_path / "court_calib.json")
    state.set_players(
        [
            {"id": 0, "name": "A", "color": "#fff", "side": 0},
            {"id": 1, "name": "B", "color": "#000", "side": 0},
            {"id": 2, "name": "C", "color": "#f00", "side": 1},
            {"id": 3, "name": "D", "color": "#00f", "side": 1},
        ],
        mode="doubles",
    )
    state.ingest(
        PlayersMessage(
            frame_id=1,
            ts=1_000_000,
            players=[
                {"track_id": track_id, "bbox": [0, 0, 10, 20], "foot_px": [5, 20], "conf": 0.9}
                for track_id in range(10, 14)
            ],
        ).to_json()
    )

    snapshot = state.snapshot()
    assert snapshot["match"]["mode"] == "doubles"
    assert len(snapshot["players"]) == 4
    assert [player["id"] for player in snapshot["players"]] == [0, 1, 2, 3]
    assert [player["side"] for player in snapshot["players"]] == [0, 0, 1, 1]


def test_doubles_court_points_use_full_doubles_width(tmp_path) -> None:
    state = FusionState(tmp_path / "court_calib.json")
    state.calibrate([[0, 0], [100, 0], [100, 100], [0, 100]])
    state.set_players(
        [
            {"id": 0, "name": "A", "color": None, "side": 0},
            {"id": 1, "name": "B", "color": None, "side": 0},
            {"id": 2, "name": "C", "color": None, "side": 1},
            {"id": 3, "name": "D", "color": None, "side": 1},
        ],
        mode="doubles",
    )
    state.ingest(
        PlayersMessage(
            frame_id=1,
            ts=1_000_000,
            players=[
                {"track_id": 10, "bbox": [0, 0, 1, 5], "foot_px": [1, 5], "conf": 0.9},
                {"track_id": 11, "bbox": [0, 0, 99, 5], "foot_px": [99, 5], "conf": 0.9},
            ],
        ).to_json()
    )

    snapshot = state.snapshot()
    # Both feet are past the net (x ~ 12.7 m), so they take the far-side slots 2 and 3.
    assert snapshot["players"][2]["court_xy_m"][1] == pytest.approx(0.061)
    assert snapshot["players"][3]["court_xy_m"][1] == pytest.approx(6.039)


def test_players_require_two_or_four_with_one_or_two_per_side(tmp_path) -> None:
    state = FusionState(tmp_path / "court_calib.json")
    with pytest.raises(ValueError, match="exactly 2"):
        state.set_players([{"id": 0, "name": "A", "color": "#fff", "side": 0}])
    with pytest.raises(ValueError, match="Each side"):
        state.set_players(
            [
                {"id": player_id, "name": str(player_id), "color": None, "side": 0}
                for player_id in range(4)
            ],
            mode="doubles",
        )
    with pytest.raises(ValueError, match="Each team"):
        state.set_players(
            [
                {
                    "id": player_id,
                    "name": str(player_id),
                    "color": None,
                    "team": 0,
                    "side": player_id,
                }
                for player_id in range(2)
            ]
        )


def test_point_confirmation_scores_only_accepted_suggestions_and_reset_clears_rally(
    tmp_path,
) -> None:
    state = FusionState(tmp_path / "court_calib.json")
    state.suggest_point(10, 0)
    state.confirm_point(10, accept=True)
    assert state.snapshot()["match"]["score"] == [1, None]

    state.suggest_point(20, 1)
    state.confirm_point(20, accept=False)
    assert state.snapshot()["match"]["score"] == [1, None]

    state.reset_rally()
    assert state.snapshot()["rally"] == {"active": False, "n": 0, "t_ms": 0, "shots": 0}
    with pytest.raises(ValueError, match="No pending"):
        state.confirm_point(20, accept=True)


def test_singles_court_points_outside_singles_sidelines_are_null(tmp_path) -> None:
    state = FusionState(tmp_path / "court_calib.json")
    state.calibrate([[0, 0], [100, 0], [100, 100], [0, 100]])
    state.set_players(
        [
            {"id": 0, "name": "A", "color": "#fff", "side": 0},
            {"id": 1, "name": "B", "color": "#000", "side": 1},
        ]
    )
    state.ingest(
        PlayersMessage(
            frame_id=1,
            ts=1_000_000,
            players=[
                {"track_id": 10, "bbox": [0, 0, 10, 10], "foot_px": [5, 1], "conf": 0.9},
                {"track_id": 11, "bbox": [0, 0, 50, 95], "foot_px": [50, 95], "conf": 0.9},
            ],
        ).to_json()
    )

    # The on-court track is matched first, so it takes its own side's slot 0.
    by_foot = {tuple(p["img_xy"]): p for p in state.snapshot()["players"]}
    assert by_foot[(5, 1)]["court_xy_m"] is None
    assert by_foot[(50, 95)]["court_xy_m"] == pytest.approx([0.67, 3.05])


def test_operator_api_and_websocket_serve_contract_state(tmp_path) -> None:
    state = FusionState(tmp_path / "court_calib.json")
    api_client = TestClient(create_operator_app(state))
    assert api_client.get("/health").status_code == 200
    assert (
        api_client.post(
            "/api/calibrate",
            json={"corners_px": [[0, 0], [100, 0], [100, 50], [0, 50]]},
        ).status_code
        == 200
    )
    assert (
        api_client.post(
            "/api/players",
            json={
                "players": [
                    {"id": 0, "name": "A", "color": "#fff", "side": 0},
                    {"id": 1, "name": "B", "color": "#000", "side": 1},
                ]
            },
        ).status_code
        == 200
    )
    assert (
        api_client.post(
            "/api/players",
            json={
                "mode": "doubles",
                "players": [
                    {"id": 0, "name": "A", "color": "#fff", "team": 0, "side": 0},
                    {"id": 1, "name": "B", "color": "#000", "team": 0, "side": 0},
                    {"id": 2, "name": "C", "color": "#f00", "team": 1, "side": 1},
                    {"id": 3, "name": "D", "color": "#00f", "team": 1, "side": 1},
                ],
            },
        ).status_code
        == 200
    )
    assert state.snapshot()["match"]["mode"] == "doubles"
    assert [player["team"] for player in state.snapshot()["players"]] == [0, 0, 1, 1]
    four_without_mode = {
        "players": [
            {"id": i, "name": str(i), "color": None, "team": i // 2, "side": i // 2}
            for i in range(4)
        ]
    }
    assert api_client.post("/api/players", json=four_without_mode).status_code == 422
    assert api_client.post("/api/players", json={"players": []}).status_code == 422
    assert api_client.post("/api/players", json={"players": [{}, {}, {}]}).status_code == 422
    assert api_client.post("/api/score", json={"side": 1, "delta": 1}).status_code == 200
    assert api_client.post("/api/score", json={"team": 0, "delta": 2}).status_code == 200
    assert state.snapshot()["match"]["score"] == [2, 1]
    assert api_client.post("/api/score", json={"team": 2, "delta": 1}).status_code == 422
    assert api_client.post("/api/swap_ids", json={"a": 0, "b": 1}).status_code == 200
    assert api_client.post("/api/calibrate", json={"corners_px": []}).status_code == 422

    with TestClient(create_ws_app(state)) as ws_client:
        with ws_client.websocket_connect("/") as websocket:
            message = websocket.receive_json()
            assert message["v"] == 1
            assert set(message) == {
                "v",
                "ts",
                "match",
                "rally",
                "players",
                "shuttle",
                "events",
                "health",
            }


def test_new_track_ids_reattach_to_nearest_player(tmp_path) -> None:
    state = FusionState(tmp_path / "court_calib.json")
    state.calibrate([[0, 0], [100, 0], [100, 100], [0, 100]])
    near = {"bbox": [0, 0, 50, 90], "foot_px": [50, 90], "conf": 0.9}
    far = {"bbox": [0, 0, 50, 10], "foot_px": [50, 10], "conf": 0.9}
    frames = [
        [{"track_id": 10, **near}, {"track_id": 11, **far}],
        # Tracker re-ids both players; new ids arrive in the opposite order.
        [{"track_id": 21, **far}, {"track_id": 20, **near}],
    ]
    for frame_id, players in enumerate(frames, start=1):
        state.ingest(
            PlayersMessage(frame_id=frame_id, ts=frame_id * 1_000_000, players=players).to_json()
        )

    snapshot = state.snapshot()
    assert [player["tracked"] for player in snapshot["players"]] == [True, True]
    assert snapshot["players"][0]["img_xy"] == [50, 90]
    assert snapshot["players"][1]["img_xy"] == [50, 10]


def _foot(x_m: float, y_m: float) -> dict[str, object]:
    # Inverse of the 100x100 px calibration below: px y=0 is the far baseline.
    px = [y_m / 6.10 * 100, (1 - x_m / 13.40) * 100]
    return {"bbox": [px[0] - 5, px[1] - 40, px[0] + 5, px[1]], "foot_px": px, "conf": 0.9}


def _square_court_state(tmp_path) -> FusionState:
    state = FusionState(tmp_path / "court_calib.json")
    state.calibrate([[0, 0], [100, 0], [100, 100], [0, 100]])
    return state


def test_doubles_new_track_reattaches_on_same_side_of_net(tmp_path) -> None:
    state = _square_court_state(tmp_path)
    state.set_players(
        [
            {"id": i, "name": str(i), "color": None, "team": i // 2, "side": i // 2}
            for i in range(4)
        ],
        mode="doubles",
    )
    state.ingest(
        PlayersMessage(
            frame_id=1,
            ts=1_000_000,
            players=[
                {"track_id": 10, **_foot(5.0, 1.0)},
                {"track_id": 11, **_foot(1.0, 4.0)},
                {"track_id": 12, **_foot(7.0, 3.0)},
                {"track_id": 13, **_foot(12.0, 2.0)},
            ],
        ).to_json()
    )
    # Tracks 10 and 12 drop; a new near-side id (beyond the 1 m net slack) appears closer to
    # the far-side slot 2 spot.
    state.ingest(
        PlayersMessage(
            frame_id=2,
            ts=2_000_000,
            players=[
                {"track_id": 11, **_foot(1.0, 4.0)},
                {"track_id": 13, **_foot(12.0, 2.0)},
                {"track_id": 20, **_foot(5.5, 3.0)},
            ],
        ).to_json()
    )

    players = state.snapshot()["players"]
    assert [p["tracked"] for p in players] == [True, True, False, True]
    assert [p["side"] for p in players] == [0, 0, 1, 1]
    assert [p["team"] for p in players] == [0, 0, 1, 1]
    assert players[0]["court_xy_m"] == pytest.approx([5.5, 3.0])
    assert players[2]["img_xy"] is None and players[2]["court_xy_m"] is None


def test_track_beyond_net_slack_never_takes_the_other_teams_slot(tmp_path) -> None:
    state = _square_court_state(tmp_path)
    state.ingest(
        PlayersMessage(
            frame_id=1, ts=1_000_000, players=[{"track_id": 10, **_foot(2.0, 3.0)}]
        ).to_json()
    )
    # Only far-side slot 1 is free; a second near-side track (an umpire, a duplicate box)
    # stays unassigned instead of becoming the far player.
    state.ingest(
        PlayersMessage(
            frame_id=2,
            ts=2_000_000,
            players=[{"track_id": 10, **_foot(2.0, 3.0)}, {"track_id": 11, **_foot(3.0, 2.0)}],
        ).to_json()
    )
    players = state.snapshot()["players"]
    assert [p["tracked"] for p in players] == [True, False]
    assert [p["side"] for p in players] == [0, 1]


def test_feet_inside_net_slack_keep_slot_and_team_side(tmp_path) -> None:
    state = _doubles_state(tmp_path)
    feet = [(3.0, 1.0), (5.5, 4.0), (10.0, 2.0), (10.0, 5.0)]
    for frame_id, x in enumerate([5.5, 6.2, 6.9, 7.3], start=1):
        feet[1] = (x, 4.0)
        state.ingest(
            PlayersMessage(
                frame_id=frame_id,
                ts=frame_id * 1_000_000,
                players=[{"track_id": 10 + i, **_foot(*xy)} for i, xy in enumerate(feet)],
            ).to_json()
        )
        if frame_id == 1:
            rusher = next(
                p["id"]
                for p in state.snapshot()["players"]
                if p["court_xy_m"] == pytest.approx([5.5, 4.0])
            )
            feet.pop()  # a far player goes undetected, so a far slot is free
    players = state.snapshot()["players"]
    assert rusher in (0, 1)
    assert players[rusher]["court_xy_m"] == pytest.approx([7.3, 4.0])
    assert [p["side"] for p in players] == [0, 0, 1, 1]
    assert [p["tracked"] for p in players].count(True) == 3


def test_end_change_by_reposting_players_moves_teams(tmp_path) -> None:
    state = _doubles_state(tmp_path)
    feet = [(3.0, 1.0), (3.0, 5.0), (10.0, 1.0), (10.0, 5.0)]
    message = PlayersMessage(
        frame_id=1,
        ts=1_000_000,
        players=[{"track_id": 10 + i, **_foot(*xy)} for i, xy in enumerate(feet)],
    )
    state.ingest(message.to_json())
    state.set_players(
        [
            {"id": i, "name": str(i), "color": None, "team": i // 2, "side": 1 - i // 2}
            for i in range(4)
        ],
        mode="doubles",
    )
    state.ingest(message.to_json())
    players = state.snapshot()["players"]
    assert [p["side"] for p in players] == [1, 1, 0, 0]
    assert [p["court_xy_m"][0] for p in players] == pytest.approx([10.0, 10.0, 3.0, 3.0])
    with pytest.raises(ValueError, match="Teammates"):
        state.set_players(
            [
                {"id": i, "name": None, "color": None, "team": i // 2, "side": i % 2}
                for i in range(4)
            ],
            mode="doubles",
        )


def _sides(state: FusionState) -> list[int]:
    return [p["side"] for p in state.snapshot()["players"]]


def test_game_end_switches_ends_and_counts_the_game(tmp_path) -> None:
    state = _doubles_state(tmp_path)
    state.add_score(1, 20)
    state.add_score(0, 20)
    state.add_score(1, 1)  # 20-21: no two-point lead yet
    assert _sides(state) == [0, 0, 1, 1]
    state.add_score(1, 1)  # 20-22 wins the game
    match = state.snapshot()["match"]
    assert (match["games"], match["score"]) == ([0, 1], [0, 0])
    assert _sides(state) == [1, 1, 0, 0]
    state._score = [29, 29]  # a long deuce, set directly
    state.suggest_point(5, None, suggested_team=0)
    state.confirm_point(5, accept=True)  # 30-29: the 30th point wins without a lead of two
    assert state.snapshot()["match"]["games"] == [1, 1]
    assert _sides(state) == [0, 0, 1, 1]


def test_deciding_game_switches_ends_once_at_eleven(tmp_path) -> None:
    state = _doubles_state(tmp_path)
    state._games = [1, 1]
    state.add_score(0, 10)
    state.add_score(1, 10)
    assert _sides(state) == [0, 0, 1, 1]
    state.add_score(1, 1)  # the leader first reaches 11
    assert _sides(state) == [1, 1, 0, 0]
    state.add_score(1, -1)  # an operator correction and the 11th point again: no re-switch
    state.add_score(1, 1)
    state.add_score(0, 1)
    assert _sides(state) == [1, 1, 0, 0]


def test_returning_track_is_matched_by_appearance_gallery(tmp_path) -> None:
    state = FusionState(
        tmp_path / "court_calib.json",
        fusion={"identity": {"gallery_s": 1.0, "app_window_frames": 2}},
    )
    state.calibrate([[0, 0], [100, 0], [100, 100], [0, 100]])
    state.set_players(
        [
            {"id": i, "name": str(i), "color": None, "team": i // 2, "side": i // 2}
            for i in range(4)
        ],
        mode="doubles",
    )
    app_a, app_b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    ts = 0
    for _ in range(40):  # 1.3 s: both far slots fill their galleries
        ts += 33_333_333
        far = [
            {"track_id": 10, **_foot(10.0, 1.0), "app": app_a},
            {"track_id": 11, **_foot(10.0, 5.0), "app": app_b},
        ]
        state.ingest(PlayersMessage(frame_id=1, ts=ts, players=far).to_json())
    a_slot = next(
        p["id"]
        for p in state.snapshot()["players"]
        if p["court_xy_m"] == pytest.approx([10.0, 1.0])
    )
    assert state._gallery[a_slot] == pytest.approx(app_a)
    for _ in range(2):  # a lighting change fills A's short running window with B's colours
        ts += 33_333_333
        far[0]["app"] = app_b
        state.ingest(PlayersMessage(frame_id=2, ts=ts, players=far).to_json())
    # Both far tracks vanish; A comes back alone, a bit closer to where B was last seen.
    ts += 1_000_000_000
    back = [{"track_id": 20, **_foot(10.0, 3.2), "app": app_a}]
    state.ingest(PlayersMessage(frame_id=3, ts=ts, players=back).to_json())
    assert state.snapshot()["players"][a_slot]["court_xy_m"] == pytest.approx([10.0, 3.2])


def _walk_state(tmp_path, positions: list[tuple[float, float]], **players: float) -> dict:
    state = FusionState(tmp_path / "court_calib.json", fusion={"players": players})
    state.calibrate([[0, 0], [100, 0], [100, 100], [0, 100]])
    for frame_id, (x_m, y_m) in enumerate(positions, start=1):
        message = PlayersMessage(
            frame_id=frame_id,
            ts=frame_id * 1_000_000_000 // 30,
            players=[{"track_id": 10, **_foot(x_m, y_m)}],
        )
        state.ingest(message.to_json())
    return state.snapshot()["players"][0]


def test_stationary_jitter_adds_no_distance_but_walking_does(tmp_path) -> None:
    import random

    rng = random.Random(4)
    # 30 s at 30 Hz standing at (3, 2) with ~5 cm foot jitter and 1.2 m single-frame flicker
    # (pose ankle vs bbox bottom), as seen for the near-left player in a 180 s replay.
    standing = [(3.0 + rng.gauss(0, 0.05), 2.0 + rng.gauss(0, 0.05)) for _ in range(900)]
    for index in range(10, 900, 40):
        standing[index] = (4.2, 2.0)
    player = _walk_state(tmp_path, standing)
    assert player["dist_m"] < 0.5
    assert player["speed_mps"] < 0.1
    # Walking 5 m toward the net (stays on side 0) at 1.5 m/s with the same jitter keeps
    # the true distance.
    walking = [(1.0 + 0.05 * i + rng.gauss(0, 0.05), 2.0 + rng.gauss(0, 0.05)) for i in range(101)]
    player = _walk_state(tmp_path, walking)
    assert player["dist_m"] == pytest.approx(5.0, abs=0.5)
    assert player["speed_mps"] == pytest.approx(1.5, abs=0.4)


def test_off_court_feet_do_not_skew_distance_or_matching(tmp_path) -> None:
    # Raw positions: no median window, so 1 m steps count exactly.
    state = FusionState(tmp_path / "court_calib.json", fusion={"players": {"median_frames": 1}})
    state.calibrate([[0, 0], [100, 0], [100, 100], [0, 100]])
    off_court = {"bbox": [0, 0, 1, 50], "foot_px": [1, 50], "conf": 0.9}  # y < 0.46 m
    frames = [
        [{"track_id": 10, **_foot(3.0, 3.0)}],
        [{"track_id": 10, **off_court}],
        [{"track_id": 10, **_foot(4.0, 3.0)}],
        [{"track_id": 10, **_foot(5.0, 3.0)}],
    ]
    for frame_id, players in enumerate(frames, start=1):
        state.ingest(
            PlayersMessage(
                frame_id=frame_id, ts=frame_id * 1_000_000_000, players=players
            ).to_json()
        )
        if frame_id == 3:
            # 1 m over 2 s from the last valid position, not a jump via (0, 0).
            player = state.snapshot()["players"][0]
            assert player["dist_m"] == pytest.approx(1.0)
            assert player["speed_mps"] == pytest.approx(0.5)
        if frame_id == 2:
            player = state.snapshot()["players"][0]
            assert player["court_xy_m"] is None and player["speed_mps"] is None
            assert player["dist_m"] is None
    player = state.snapshot()["players"][0]
    assert player["dist_m"] == pytest.approx(2.0)
    assert player["speed_mps"] == pytest.approx(0.35 * 1.0 + 0.65 * 0.5)  # EMA still smooths
    assert state._player_history[0][1:] == pytest.approx((5.0, 3.0))


def test_players_message_with_appearance_vector_is_accepted(tmp_path) -> None:
    state = _square_court_state(tmp_path)
    state.ingest(
        json.dumps(
            {
                "v": 1,
                "type": "players",
                "frame_id": 1,
                "ts": 1_000_000,
                "players": [{"track_id": 10, **_foot(2.0, 3.0), "app": [0.1, 0.5, 0.3]}],
            }
        )
    )
    assert state.snapshot()["players"][0]["tracked"] is True


def test_swap_ids_moves_tracks_but_names_stay_with_ids(tmp_path) -> None:
    state = _square_court_state(tmp_path)
    state.set_players(
        [
            {"id": 0, "name": "A", "color": "#fff", "team": 0, "side": 0},
            {"id": 1, "name": "B", "color": "#000", "team": 1, "side": 1},
        ]
    )
    near = {"track_id": 10, **_foot(2.0, 3.0)}
    far = {"track_id": 11, **_foot(11.0, 3.0)}
    state.ingest(PlayersMessage(frame_id=1, ts=1_000_000, players=[near, far]).to_json())
    state.swap_ids(0, 1)
    swapped = state.snapshot()["players"]
    assert swapped[1]["img_xy"] == near["foot_px"]
    assert (swapped[1]["name"], swapped[1]["color"], swapped[1]["team"]) == ("B", "#000", 1)
    assert (swapped[0]["name"], swapped[0]["img_xy"]) == ("A", far["foot_px"])
    # A cross-team swap is a correction, not an end change: team sides stay, so the next
    # frame puts each track back on the team that plays from its side.
    assert [p["side"] for p in swapped] == [0, 1]
    state.ingest(PlayersMessage(frame_id=2, ts=2_000_000, players=[near, far]).to_json())
    players = state.snapshot()["players"]
    assert (players[0]["name"], players[0]["img_xy"]) == ("A", near["foot_px"])
    assert (players[1]["name"], players[1]["img_xy"]) == ("B", far["foot_px"])


def test_shuttle_has_no_court_position_while_flying(tmp_path) -> None:
    state = _square_court_state(tmp_path)
    state.ingest(ShuttleMessage(frame_id=1, ts=1_000_000, visible=True, px=[50, 50]).to_json())
    shuttle = state.snapshot()["shuttle"]
    assert shuttle["img_xy"] == [50, 50]
    assert shuttle["landing_xy_m_est"] is None
    assert "court_xy_m" not in shuttle


def _doubles_state(tmp_path) -> FusionState:
    state = _square_court_state(tmp_path)
    state.set_players(
        [
            {"id": i, "name": str(i), "color": None, "team": i // 2, "side": i // 2}
            for i in range(4)
        ],
        mode="doubles",
    )
    return state


# Near-side partners walk through each other; the tracker swaps their ids at the crossing.
_A_PATH = [1.0, 2.0, 2.8, 3.3, 4.0, 5.0]
_B_PATH = [5.0, 4.0, 3.2, 2.7, 2.0, 1.0]


def _cross(state: FusionState, app_a: list[float] | None, app_b: list[float] | None) -> None:
    for frame_id, (a_y, b_y) in enumerate(zip(_A_PATH, _B_PATH, strict=True), start=1):
        a_track, b_track = (10, 11) if frame_id <= 3 else (11, 10)
        players = [
            {"track_id": a_track, **_foot(3.0, a_y), "app": app_a},
            {"track_id": b_track, **_foot(3.0, b_y), "app": app_b},
        ]
        state.ingest(
            PlayersMessage(frame_id=frame_id, ts=frame_id * 1_000_000, players=players).to_json()
        )


def test_partners_crossing_keep_ids_by_appearance(tmp_path) -> None:
    state = _doubles_state(tmp_path)
    _cross(state, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    players = state.snapshot()["players"]
    assert players[0]["court_xy_m"] == pytest.approx([3.0, 5.0])
    assert players[1]["court_xy_m"] == pytest.approx([3.0, 1.0])
    assert [p["side"] for p in players[:2]] == [0, 0]


def test_null_appearance_falls_back_to_position(tmp_path) -> None:
    state = _doubles_state(tmp_path)
    _cross(state, None, None)
    # Without appearance the nearest-position match follows the crossing detections.
    players = state.snapshot()["players"]
    assert players[0]["court_xy_m"] == pytest.approx([3.0, 1.0])
    assert players[1]["court_xy_m"] == pytest.approx([3.0, 5.0])


def test_operator_swap_sticks_against_appearance(tmp_path) -> None:
    state = _doubles_state(tmp_path)
    app_a, app_b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    _cross(state, app_a, app_b)
    state.swap_ids(0, 1)
    for frame_id, (a_y, b_y) in enumerate([(4.8, 1.2), (4.6, 1.4)], start=7):
        players = [
            {"track_id": 11, **_foot(3.0, a_y), "app": app_a},
            {"track_id": 10, **_foot(3.0, b_y), "app": app_b},
        ]
        state.ingest(
            PlayersMessage(frame_id=frame_id, ts=frame_id * 1_000_000, players=players).to_json()
        )
    players = state.snapshot()["players"]
    assert players[1]["court_xy_m"] == pytest.approx([3.0, 4.6])
    assert players[0]["court_xy_m"] == pytest.approx([3.0, 1.4])


def test_min_cost_pairs_handles_rectangular_costs() -> None:
    assert _min_cost_pairs([[5.0, 1.0], [1.0, 5.0]]) == [(0, 1), (1, 0)]
    assert _min_cost_pairs([[3.0], [1.0], [2.0]]) == [(1, 0)]
    assert _min_cost_pairs([]) == []


def test_eight_detections_fill_four_slots_by_side_and_position(tmp_path) -> None:
    state = _doubles_state(tmp_path)
    players_xy = [(3.0, 1.0), (3.0, 5.0), (10.0, 1.0), (10.0, 5.0)]
    others = [(0.5, 3.0), (6.0, 0.2), (7.5, 5.9), (13.0, 3.0)]  # umpire, spectators in ROI
    first = [{"track_id": i, **_foot(*xy)} for i, xy in enumerate(players_xy)]
    state.ingest(PlayersMessage(frame_id=1, ts=1_000_000, players=first).to_json())
    # Frame 1 has no history, so partners may take either slot of their side.
    start = [p["court_xy_m"] for p in state.snapshot()["players"]]
    detections = [{"track_id": 20 + i, **_foot(*xy)} for i, xy in enumerate(others)]
    detections += [{"track_id": 30 + i, **_foot(x + 0.1, y)} for i, (x, y) in enumerate(players_xy)]
    state.ingest(PlayersMessage(frame_id=2, ts=2_000_000, players=detections).to_json())
    players = state.snapshot()["players"]
    assert [p["side"] for p in players] == [0, 0, 1, 1]
    for slot, (x, y) in enumerate(start):
        assert players[slot]["court_xy_m"] == pytest.approx([x + 0.1, y])


def _doubles_after_end_change(tmp_path) -> FusionState:
    # Team 0 now plays from the far side (side 1), team 1 from the near side.
    state = _square_court_state(tmp_path)
    state.set_players(
        [
            {"id": i, "name": str(i), "color": None, "team": i // 2, "side": 1 - i // 2}
            for i in range(4)
        ],
        mode="doubles",
    )
    return state


def test_confirm_point_maps_suggested_side_to_team_after_end_change(tmp_path) -> None:
    state = _doubles_after_end_change(tmp_path)
    state.suggest_point(10, 0)  # near side won; that is team 1 now
    state.confirm_point(10, accept=True)
    state.suggest_point(20, 0, suggested_team=0)  # an explicit team wins over the side
    state.confirm_point(20, accept=True)
    assert state.snapshot()["match"]["score"] == [1, 1]


_FRAME_NS = 16_666_667


def _two_shot_rally(state: FusionState) -> tuple[int, list[dict]]:
    """Player 2 (near) hits it up, player 0 (far) returns it; last seen at px [16, 40]."""
    feet = [(10.0, 1.0), (10.0, 5.0), (3.0, 1.0), (3.0, 5.0)]
    state.ingest(
        PlayersMessage(
            frame_id=1,
            ts=1_000_000,
            players=[{"track_id": 10 + i, **_foot(*xy)} for i, xy in enumerate(feet)],
        ).to_json()
    )
    ts = 1_000_000_000
    path_y = [50 + 2 * i for i in range(10)] + [66 - 2 * i for i in range(24)]
    path_y += [22 + 2 * i for i in range(10)]
    events: list[dict] = []
    for y in path_y:
        ts += _FRAME_NS
        state.ingest(ShuttleMessage(frame_id=2, ts=ts, visible=True, px=[16.0, y]).to_json())
        events += state.snapshot()["events"]
    return ts, events


def _feed(state: FusionState, ts: int, frames: int, px: list[float] | None) -> tuple[int, list]:
    events: list[dict] = []
    for _ in range(frames):
        ts += _FRAME_NS
        state.ingest(ShuttleMessage(frame_id=3, ts=ts, visible=px is not None, px=px).to_json())
        events += state.snapshot()["events"]
    return ts, events


def test_shuttle_and_hits_drive_rally_shots_and_point_suggestion(tmp_path) -> None:
    state = _doubles_after_end_change(tmp_path)
    ts, events = _two_shot_rally(state)
    snapshot = state.snapshot()
    assert snapshot["rally"]["active"] is True
    assert snapshot["rally"]["shots"] == 2
    assert [e["data"]["by"] for e in events if e["kind"] == "shot"] == [2, 0]
    # last_shot is the answered shot (player 2's), whose speed is known once returned.
    assert snapshot["shuttle"]["last_shot"]["by"] == 2
    assert snapshot["shuttle"]["landing_xy_m_est"] is None  # still flying
    ts, lost_events = _feed(state, ts, 130, None)  # > 1.8 s unseen ends the rally
    events += lost_events
    snapshot = state.snapshot()
    assert snapshot["shuttle"]["status"] == "lost"
    assert snapshot["rally"]["active"] is False
    assert [e["kind"] for e in events] == [
        "rally_start",
        "shot",
        "shot",
        "rally_end",
        "point_suggested",
    ]
    point = events[-1]
    # Landed in on the far half (last seen at px y=40 -> x ~ 8 m): side 0, i.e. team 1.
    assert point["data"]["suggested_side"] == 0 and point["data"]["suggested_team"] == 1
    assert snapshot["shuttle"]["landing_xy_m_est"] == point["data"]["landing_xy_m_est"]
    state.confirm_point(point["ts"], accept=True)
    assert state.snapshot()["match"]["score"] == [None, 1]


_REST_PX = [16.0, 80.0]  # near half, x ~ 2.68 m: the near side (1) loses instead


def test_shuttle_resting_on_floor_ends_rally_with_rest_point_as_landing(tmp_path) -> None:
    state = _doubles_after_end_change(tmp_path)
    ts, events = _two_shot_rally(state)
    # It drops and lies still (detections jitter by 4 px): tracked for rest_ms, then 'lost'
    # until end_gap_ms runs out. The landing is the rest anchor, not the last jittered sighting.
    for px, frames in ((_REST_PX, 1), ([16.0, 84.0], 149)):
        ts, rest_events = _feed(state, ts, frames, px)
        events += rest_events
    snapshot = state.snapshot()
    assert snapshot["rally"]["active"] is False
    kinds = [e["kind"] for e in events]
    assert kinds == ["rally_start", "shot", "shot", "rally_end", "point_suggested"]
    expected = [round(v, 2) for v in transform_point(state.homography, _REST_PX)]
    point = events[-1]["data"]
    assert point["landing_xy_m_est"] == expected
    assert point["suggested_side"] == 1 and point["in_bounds"] is True
    assert snapshot["shuttle"]["landing_xy_m_est"] == expected


def test_rest_point_after_rally_end_replaces_pending_suggestion(tmp_path) -> None:
    state = _doubles_after_end_change(tmp_path)
    ts, _ = _two_shot_rally(state)
    ts, events = _feed(state, ts, 130, None)
    first = events[-1]
    assert first["kind"] == "point_suggested" and first["data"]["suggested_side"] == 0
    ts, events = _feed(state, ts, 40, _REST_PX)  # seen at rest within rest_landing_ms
    assert [e["kind"] for e in events] == ["point_suggested"]
    assert events[0]["ts"] == first["ts"] and events[0]["data"]["suggested_side"] == 1
    assert list(state._pending_points) == [first["ts"]]
    assert state._pending_points[first["ts"]]["suggested_side"] == 1
    state.confirm_point(first["ts"], accept=True)
    assert state.snapshot()["match"]["score"] == [1, None]  # side 1 is team 0 now


def test_rest_point_after_operator_answered_does_not_reopen_suggestion(tmp_path) -> None:
    state = _doubles_after_end_change(tmp_path)
    ts, _ = _two_shot_rally(state)
    ts, events = _feed(state, ts, 130, None)
    point_ts = events[-1]["ts"]
    state.confirm_point(point_ts, accept=False)
    ts, events = _feed(state, ts, 40, _REST_PX)
    assert events == [] and state._pending_points == {}
    with pytest.raises(ValueError):
        state.confirm_point(point_ts, accept=True)


def test_slow_ws_client_keeps_events_from_dropped_frames() -> None:
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=1)
    point = {"ts": 1, "kind": "point_suggested", "data": {}}
    shot = {"ts": 2, "kind": "shot", "data": {}}
    offer_latest(queue, {"ts": 1, "events": [point]})
    offer_latest(queue, {"ts": 2, "events": [shot]})
    offer_latest(queue, {"ts": 3, "events": []})
    assert queue.get_nowait() == {"ts": 3, "events": [point, shot]}


def test_fusion_tunables_come_from_config(tmp_path) -> None:
    state = FusionState(
        tmp_path / "court_calib.json",
        fusion={
            "identity": {"app_window_frames": 5, "side_mismatch_cost": 50},
            "rally": {"end_gap_ms": 500, "shot_match_window_ms": 80, "rest_landing_ms": 2000},
            "shuttle": {
                "max_px_per_s": 3000,
                "predict_ms": 100,
                "direction_max_gap_ms": 300,
                "rest_radius_px": 5,
                "rest_ms": 600,
            },
            "players": {"median_frames": 3, "deadband_m": 0.2},
        },
    )
    assert state._kf.predict_ns == 100_000_000
    assert (state._median_frames, state._deadband_m) == (3, 0.2)
    assert (state._app_window_frames, state._app_weight_m, state._side_mismatch_cost) == (
        5,
        2.0,
        50,
    )
    assert state._fsm.end_gap_ns == 500_000_000
    assert state._shots.match_window_ns == 80_000_000 and state._shots.refractory_ns == 200_000_000
    assert state._kf.max_px_per_s == 3000
    assert state._kf.direction_max_gap_ns == 300_000_000
    assert (state._kf.rest_radius_px, state._kf.rest_ns) == (5, 600_000_000)
    assert state._fsm.rest_landing_ns == 2_000_000_000
    assert FusionState(tmp_path / "c.json", fusion={"shuttle": None})._kf.max_px_per_s is None


def test_health_fps_is_capture_rate_and_dropped_is_slowest_worker_skips(tmp_path) -> None:
    state = FusionState(tmp_path / "court_calib.json")
    assert state.snapshot()["health"]["fps"] is None
    # Capture at 60 fps (ts step 1/60 s); players sees every 2nd frame, shuttle every frame.
    for frame_id in range(0, 61):
        ts = 1_000_000_000 + frame_id * 1_000_000_000 // 60
        state.ingest(ShuttleMessage(frame_id=frame_id, ts=ts).to_json())
        if frame_id % 2 == 0:
            state.ingest(PlayersMessage(frame_id=frame_id, ts=ts, players=[]).to_json())
    health = state.snapshot()["health"]
    assert health["fps"] == pytest.approx(60.0, rel=1e-3)
    assert health["dropped"] == 30
    # Capture restart: frame ids start over at a later capture ts.
    state.ingest(ShuttleMessage(frame_id=0, ts=10_000_000_000).to_json())
    state.ingest(ShuttleMessage(frame_id=30, ts=11_000_000_000).to_json())
    assert state.snapshot()["health"]["fps"] == pytest.approx(30.0)


def test_run_at_rate_holds_30_hz_on_a_fake_clock_and_skips_when_late() -> None:
    now = [100.0]
    ticks: list[float] = []

    def tick() -> None:
        ticks.append(now[0])
        # Each tick costs 5 ms; tick 10 stalls 100 ms (three periods).
        now[0] += 0.1 if len(ticks) == 10 else 0.005

    async def sleep(seconds: float) -> None:
        assert seconds >= 0
        now[0] += seconds + 0.012  # every wake-up is 12 ms late (Windows timer granularity)
        if len(ticks) >= 100:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_at_rate(tick, 30, clock=lambda: now[0], sleep=sleep))
    period = 1 / 30
    slots = [round((t - 100.0) / period, 6) for t in ticks]
    # Ticks are never early, stay within the wake-up lateness of the grid, never drift ...
    assert all(0 <= s - round(s) < 0.012 / period + 1e-6 for s in slots)
    # ... and the stall skips the missed slots instead of bursting to catch up.
    assert [round(s) for s in slots[:12]] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14]
    assert round(slots[-1]) == 102  # 100 ticks: slots 0-9, then 13-102


def test_ingest_connects_to_configured_worker_endpoints(tmp_path, monkeypatch) -> None:
    import zmq.asyncio

    from hud.config import AppConfig, BusConfig

    connected: list[str] = []
    real_socket = zmq.asyncio.Context.socket

    def socket(self, kind):
        sock = real_socket(self, kind)
        sock.connect = connected.append
        return sock

    monkeypatch.setattr(zmq.asyncio.Context, "socket", socket)
    config = AppConfig(bus=BusConfig(endpoints={"players": "tcp://127.0.0.1:59001"}))

    async def run() -> None:
        task = asyncio.create_task(ingest_worker_messages(FusionState(tmp_path / "c.json"), config))
        await asyncio.sleep(0.05)
        task.cancel()

    asyncio.run(run())
    assert connected[0] == "tcp://127.0.0.1:59001"
    assert connected[1:] and all(not e.startswith("tcp://127.0.0.1:59001") for e in connected[1:])


def test_serve_builds_state_with_configured_calibration_path(tmp_path, monkeypatch) -> None:
    import uvicorn

    import hud.server.ws_server as ws_server
    from hud.server import operator_api

    calibration = tmp_path / "calib" / "court.json"
    config_path = tmp_path / "hud.yaml"
    config_path.write_text("fusion:\n  calibration_path: calib/court.json\n", encoding="utf-8")
    seen: dict[str, FusionState] = {}

    async def fake_ingest(state, _config) -> None:
        seen["state"] = state

    async def fake_serve(self) -> None:
        return None

    monkeypatch.setattr(ws_server, "ingest_worker_messages", fake_ingest)
    monkeypatch.setattr(uvicorn.Server, "serve", fake_serve)
    asyncio.run(operator_api.serve(str(config_path)))
    assert seen["state"].calibration_path == calibration
