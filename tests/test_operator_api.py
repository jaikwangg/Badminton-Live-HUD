from __future__ import annotations

import uuid

import cv2
import numpy as np
from fastapi.testclient import TestClient
from httpx import Response

from hud.bus.shm_ring import SharedFrameRing
from hud.config import AppConfig
from hud.fusion.state import FusionState
from hud.server.operator_api import create_operator_app


def test_frame_jpg_serves_latest_bus_frame_at_source_size(tmp_path) -> None:
    config = AppConfig()
    config.bus.name = f"hud_test_{uuid.uuid4().hex[:12]}"
    config.source.width, config.source.height = 64, 48
    client = TestClient(create_operator_app(FusionState(tmp_path / "court_calib.json"), config))
    assert client.get("/api/frame.jpg").status_code == 503  # no bus yet

    ring = SharedFrameRing(config.bus.name, 64, 48)
    try:
        assert client.get("/api/frame.jpg").status_code == 503  # bus but no frame
        ring.write(np.zeros((48, 64, 3), dtype=np.uint8), 1)
        ring.write(np.full((48, 64, 3), 200, dtype=np.uint8), 2)
        response = client.get("/api/frame.jpg")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (48, 64, 3)
        assert abs(int(image.mean()) - 200) <= 2  # newest frame, not the first
        # The request's read-only attach must not break the writer's ring.
        ring.write(np.zeros((48, 64, 3), dtype=np.uint8), 3)
    finally:
        ring.close()
        ring.unlink()


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


def test_overlay_page_is_served_next_to_operator_page(tmp_path) -> None:
    client = TestClient(create_operator_app(FusionState(tmp_path / "court_calib.json")))
    overlay = client.get("/overlay/")
    assert overlay.status_code == 200
    assert "overlay.js" in overlay.text
    assert client.get("/overlay/overlay.js").status_code == 200
    # Without the slash the relative overlay.js/css would resolve to /, so it must redirect.
    assert client.get("/overlay", follow_redirects=False).status_code in (301, 307)
    assert "overlay.js" not in client.get("/").text
    assert client.get("/health").status_code == 200
