from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from hud.bus.messages import HeartbeatMessage, PlayersMessage
from hud.fusion.homography import compute_homography, transform_point
from hud.fusion.state import FusionState, validate_game_state
from hud.server.operator_api import create_operator_app
from hud.server.ws_server import create_ws_app


def test_homography_maps_reference_corners_within_one_centimeter() -> None:
    corners = [[100, 50], [900, 50], [900, 450], [100, 450]]
    transform = compute_homography(corners)
    point = transform_point(transform, [500, 250])
    assert point == pytest.approx([6.7, 3.05], abs=0.01)


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
        ]
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


def test_players_require_two_or_four_with_one_or_two_per_side() -> None:
    state = FusionState()
    with pytest.raises(ValueError, match="two or four"):
        state.set_players([{"id": 0, "name": "A", "color": "#fff", "side": 0}])
    with pytest.raises(ValueError, match="Each side"):
        state.set_players(
            [
                {"id": player_id, "name": str(player_id), "color": None, "side": 0}
                for player_id in range(4)
            ]
        )


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
                {"track_id": 11, "bbox": [0, 0, 10, 50], "foot_px": [5, 50], "conf": 0.9},
            ],
        ).to_json()
    )

    snapshot = state.snapshot()
    assert snapshot["players"][0]["court_xy_m"] is None
    assert snapshot["players"][1]["court_xy_m"] == pytest.approx([0.67, 3.05])


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
                "players": [
                    {"id": 0, "name": "A", "color": "#fff", "side": 0},
                    {"id": 1, "name": "B", "color": "#000", "side": 0},
                    {"id": 2, "name": "C", "color": "#f00", "side": 1},
                    {"id": 3, "name": "D", "color": "#00f", "side": 1},
                ]
            },
        ).status_code
        == 200
    )
    assert api_client.post("/api/players", json={"players": []}).status_code == 422
    assert api_client.post("/api/score", json={"side": 1, "delta": 1}).status_code == 200
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
