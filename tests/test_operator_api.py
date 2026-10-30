from __future__ import annotations

from fastapi.testclient import TestClient
from httpx import Response

from hud.fusion.state import FusionState
from hud.server.operator_api import create_operator_app


def test_point_confirm_and_rally_reset_endpoints(tmp_path) -> None:
    state = FusionState(tmp_path / "court_calib.json")
    client = TestClient(create_operator_app(state))

    def confirm(body: dict) -> Response:
        return client.post("/api/point/confirm", json=body)

    state.suggest_point(10, 1)
    assert confirm({"event_ts": 10, "accept": True}).status_code == 200
    assert state.snapshot()["match"]["score"] == [None, 1]
    # Already consumed, wrong types and missing fields are all rejected.
    assert confirm({"event_ts": 10, "accept": True}).status_code == 422
    assert confirm({"event_ts": "10", "accept": True}).status_code == 422
    assert confirm({"event_ts": 10, "accept": 1}).status_code == 422
    assert confirm({}).status_code == 422

    state.suggest_point(20, 0)
    assert client.post("/api/rally/reset", json={}).status_code == 200
    assert state.snapshot()["rally"] == {"active": False, "n": 0, "t_ms": 0, "shots": 0}
    assert confirm({"event_ts": 20, "accept": True}).status_code == 422


def test_operator_page_is_served_without_shadowing_api(tmp_path) -> None:
    client = TestClient(create_operator_app(FusionState(tmp_path / "court_calib.json")))
    page = client.get("/")
    assert page.status_code == 200
    assert 'src="operator.js"' in page.text
    assert client.get("/operator.js").status_code == 200
    assert client.get("/health").json()["workers"]
