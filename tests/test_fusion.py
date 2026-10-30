from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from hud.bus.messages import HeartbeatMessage, PlayersMessage, ShuttleMessage
from hud.fusion.homography import compute_homography, transform_point
from hud.fusion.state import FusionState, _min_cost_pairs, validate_game_state
from hud.server.operator_api import create_operator_app
from hud.server.ws_server import create_ws_app


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


def test_worker_without_recent_heartbeat_is_reported_down() -> None:
    state = FusionState()
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


def test_players_require_two_or_four_with_one_or_two_per_side() -> None:
    state = FusionState()
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


def test_point_confirmation_scores_only_accepted_suggestions_and_reset_clears_rally() -> None:
    state = FusionState()
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
    # Tracks 10 and 12 drop; a new near-side id appears closer to the far-side slot 2 spot.
    state.ingest(
        PlayersMessage(
            frame_id=2,
            ts=2_000_000,
            players=[
                {"track_id": 11, **_foot(1.0, 4.0)},
                {"track_id": 13, **_foot(12.0, 2.0)},
                {"track_id": 20, **_foot(6.5, 3.0)},
            ],
        ).to_json()
    )

    players = state.snapshot()["players"]
    assert [p["tracked"] for p in players] == [True, True, False, True]
    assert [p["side"] for p in players] == [0, 0, 1, 1]
    assert [p["team"] for p in players] == [0, 0, 1, 1]
    assert players[0]["court_xy_m"] == pytest.approx([6.5, 3.0])
    assert players[2]["img_xy"] is None and players[2]["court_xy_m"] is None


def test_new_track_falls_back_to_other_side_when_no_slot_is_free(tmp_path) -> None:
    state = _square_court_state(tmp_path)
    state.ingest(
        PlayersMessage(
            frame_id=1, ts=1_000_000, players=[{"track_id": 10, **_foot(2.0, 3.0)}]
        ).to_json()
    )
    # Only far-side slot 1 is free, so the second near-side track still gets it.
    state.ingest(
        PlayersMessage(
            frame_id=2,
            ts=2_000_000,
            players=[{"track_id": 10, **_foot(2.0, 3.0)}, {"track_id": 11, **_foot(3.0, 2.0)}],
        ).to_json()
    )
    players = state.snapshot()["players"]
    assert [p["tracked"] for p in players] == [True, True]
    assert [p["side"] for p in players] == [0, 0]


def test_off_court_feet_do_not_skew_distance_or_matching(tmp_path) -> None:
    state = _square_court_state(tmp_path)
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
    # The next frame keeps feeding track 10 into id 1.
    state.ingest(PlayersMessage(frame_id=2, ts=2_000_000, players=[near, far]).to_json())
    players = state.snapshot()["players"]
    assert (players[1]["name"], players[1]["img_xy"]) == ("B", near["foot_px"])
    assert (players[0]["name"], players[0]["img_xy"]) == ("A", far["foot_px"])


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
