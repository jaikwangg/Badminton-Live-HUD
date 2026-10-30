from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from hud.bus.messages import PlayersMessage
from hud.config import load_config
from hud.models.detector import Det, YoloByteTrackDetector, create_detector
from hud.models.detector_app import chest_appearance
from hud.workers import players as players_worker
from hud.workers.players import PlayersWorker


class FakeDetector:
    def infer(self, img: object) -> list[Det]:
        assert img.shape == (100, 200, 3)
        return [
            Det([20, 10, 60, 80], 0.9, 4),
            Det([120, 10, 180, 90], 0.8, 5),
            Det([80, 10, 100, 50], 0.7, None),
        ]


class FakePublisher:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, message: str) -> None:
        self.messages.append(message)


def test_worker_emits_feet_ids_and_contract_version() -> None:
    publisher = FakePublisher()
    worker = PlayersWorker(FakeDetector(), publisher, roi=[0.0, 0.0, 0.7, 1.0])
    worker.process(7, 1234, 200, 100, bytes(200 * 100 * 3))
    players = PlayersMessage.from_json(publisher.messages[0])
    assert players.v == 1
    assert players.frame_id == 7 and players.ts == 1234
    assert len(players.players) == 1
    assert players.players[0].track_id == 4
    assert players.players[0].bbox == [20, 10, 60, 80]
    assert players.players[0].foot_px == [40.0, 80]
    heartbeat = json.loads(publisher.messages[1])
    assert heartbeat["v"] == 1
    assert heartbeat["type"] == "heartbeat"
    assert heartbeat["worker"] == "players"
    assert heartbeat["proc_ms"] is not None


def test_heartbeat_is_printed_for_supervisor(capsys) -> None:
    worker = PlayersWorker(FakeDetector(), FakePublisher())
    worker.process(7, 1234, 200, 100, bytes(200 * 100 * 3))
    line = capsys.readouterr().out.strip()
    assert json.loads(line)["worker"] == "players"


def test_worker_forwards_four_tracked_players() -> None:
    class FourPlayerDetector:
        def infer(self, img: object) -> list[Det]:
            assert img.shape == (100, 200, 3)
            return [Det([10 * i, 10, 10 * i + 8, 80], 0.9, i) for i in range(4)]

    publisher = FakePublisher()
    worker = PlayersWorker(FourPlayerDetector(), publisher)
    worker.process(12, 5678, 200, 100, bytes(200 * 100 * 3))

    message = PlayersMessage.from_json(publisher.messages[0])
    assert [player.track_id for player in message.players] == [0, 1, 2, 3]


APP_CFG = yaml.safe_load((Path(__file__).parents[1] / "config" / "default.yaml").read_text())[
    "models"
]["players"]["app"]
RED, FLOOR_BLUE, WHITE, BLACK = (0, 0, 200), (200, 60, 20), (255, 255, 255), (0, 0, 0)


def _ankles(left: list[float], right: list[float]) -> list[list[float]]:
    return [[0.0, 0.0, 0.0]] * 15 + [left, right]


def _worker(detections: list[Det], **kwargs: object) -> tuple[PlayersWorker, FakePublisher]:
    class ListDetector:
        def infer(self, img: object) -> list[Det]:
            return detections

    publisher = FakePublisher()
    return PlayersWorker(ListDetector(), publisher, **kwargs), publisher


def test_foot_uses_confident_ankles_and_falls_back_to_bbox() -> None:
    detections = [
        Det([0, 0, 40, 80], 0.9, 1, _ankles([10, 70, 0.9], [20, 74, 0.8])),
        Det([50, 0, 90, 80], 0.9, 2, _ankles([60, 70, 0.9], [80, 74, 0.1])),
        Det([100, 0, 140, 80], 0.9, 3, _ankles([110, 70, 0.1], [130, 74, 0.1])),
        Det([150, 0, 190, 80], 0.9, 4),
    ]
    worker, publisher = _worker(detections, ankle_confidence=0.5)
    worker.process(1, 1, 200, 100, bytes(200 * 100 * 3))
    feet = [p.foot_px for p in PlayersMessage.from_json(publisher.messages[0]).players]
    assert feet == [[15.0, 80], [60.0, 80], [120.0, 80], [170.0, 80]]


def test_foot_uses_bbox_bottom_for_far_player_and_ankles_when_bbox_clipped() -> None:
    detections = [
        # Far player: ankles 16 px above the bbox bottom; ground contact is the bottom.
        Det([100, 20, 110, 60], 0.9, 1, _ankles([103, 44, 0.9], [107, 44, 0.9])),
        # Clipped bbox: confident ankles below its bottom edge win.
        Det([0, 0, 40, 80], 0.9, 2, _ankles([10, 90, 0.9], [20, 94, 0.9])),
    ]
    worker, publisher = _worker(detections, ankle_confidence=0.5)
    worker.process(1, 1, 200, 100, bytes(200 * 100 * 3))
    feet = [p.foot_px for p in PlayersMessage.from_json(publisher.messages[0]).players]
    assert feet == [[105.0, 60], [15.0, 92.0]]


def test_bad_frames_are_skipped_until_failures_persist(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clock = [100.0]
    monkeypatch.setattr(players_worker.time, "monotonic", lambda: clock[0])
    worker = PlayersWorker(FakeDetector(), FakePublisher())
    bad = (1, 1, 0, 0, b"")  # shape (0, 0, 3) fails FakeDetector's shape assert
    worker.process_or_skip(*bad)
    clock[0] += 2.0
    worker.process_or_skip(*bad)
    out = capsys.readouterr()
    assert out.err.count("skipped frame") == 1
    assert json.loads(out.out.strip().splitlines()[-1])["worker"] == "players"
    worker.process_or_skip(2, 2, 200, 100, bytes(200 * 100 * 3))  # success resets the streak
    clock[0] += 10.0
    worker.process_or_skip(*bad)
    clock[0] += players_worker.MAX_FAILING_S + 0.1
    with pytest.raises(AssertionError):
        worker.process_or_skip(*bad)


def test_app_is_null_without_app_config() -> None:
    worker, publisher = _worker([Det([0, 0, 40, 80], 0.9, 1)])
    worker.process(1, 1, 200, 100, bytes(200 * 100 * 3))
    assert PlayersMessage.from_json(publisher.messages[0]).players[0].app is None


def test_unknown_app_method_is_rejected() -> None:
    with pytest.raises(ValueError):
        _worker([], app={**APP_CFG, "method": "reid"})


def _image(fill: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:] = fill
    return image


def test_chest_appearance_removes_floor_pixels() -> None:
    shirt_only = chest_appearance(_image(RED), [0, 0, 100, 100], APP_CFG)
    mixed = _image(RED)
    mixed[:, 50:] = FLOOR_BLUE
    with_floor = chest_appearance(mixed, [0, 0, 100, 100], APP_CFG)
    assert shirt_only is not None and len(shirt_only) == APP_CFG["length"]
    assert shirt_only == with_floor
    assert shirt_only[0] == 1.0 and sum(shirt_only) == pytest.approx(1.0)


def test_chest_appearance_dark_light_and_null_cases() -> None:
    assert chest_appearance(_image(WHITE), [0, 0, 100, 100], APP_CFG)[-1] == 1.0
    assert chest_appearance(_image(BLACK), [0, 0, 100, 100], APP_CFG)[-2] == 1.0
    assert chest_appearance(_image(FLOOR_BLUE), [0, 0, 100, 100], APP_CFG) is None
    assert chest_appearance(_image(RED), [10, 10, 11, 11], APP_CFG) is None


def test_four_players_each_get_their_own_app() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    colours = [RED, WHITE, BLACK, (0, 200, 0)]
    for i, colour in enumerate(colours):
        image[:, 50 * i : 50 * i + 50] = colour
    detections = [Det([50 * i, 0, 50 * i + 50, 100], 0.9, i) for i in range(4)]
    worker, publisher = _worker(detections, app=APP_CFG)
    worker.process(3, 9, 200, 100, image.tobytes())
    players = PlayersMessage.from_json(publisher.messages[0]).players
    apps = [player.app for player in players]
    assert [player.track_id for player in players] == [0, 1, 2, 3]
    assert all(app is not None and len(app) == APP_CFG["length"] for app in apps)
    assert len({tuple(app) for app in apps}) == 4


def test_run_binds_the_configured_bus_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config()
    config.bus.endpoints["players"] = "tcp://127.0.0.1:59999"
    bound: list[str] = []

    def fake_publisher(address: str) -> None:
        bound.append(address)
        raise RuntimeError("stop")

    monkeypatch.setattr(players_worker, "create_detector", lambda _cfg: None)
    monkeypatch.setattr(players_worker, "ZmqPublisher", fake_publisher)
    with pytest.raises(RuntimeError, match="stop"):
        players_worker.run(config)
    assert bound == ["tcp://127.0.0.1:59999"]


class _Tensor(list):
    def cpu(self) -> _Tensor:
        return self

    def tolist(self) -> list:
        return list(self)


class _Result:
    def __init__(self, rows: list[list[float]]) -> None:
        # rows: [x1, y1, x2, y2, conf] in crop pixels; one keypoint at the box bottom-left.
        self.boxes = SimpleNamespace(
            xyxy=_Tensor(row[:4] for row in rows), conf=_Tensor(row[4] for row in rows)
        )
        self.keypoints = SimpleNamespace(data=_Tensor([[row[0], row[3], 0.9]] for row in rows))


class _FakeYolo:
    """Stand-in for ultralytics.YOLO that answers each crop by its (height, width)."""

    def __init__(self, path: str) -> None:
        self.calls: list[tuple[str, list[tuple[int, int]]]] = []
        self.answers: dict[tuple[int, int], list[list[float]]] = {}

    def predict(self, images: list[np.ndarray], **kwargs: object) -> list[_Result]:
        self.calls.append(("predict", [img.shape[:2] for img in images]))
        return [_Result(self.answers.get(img.shape[:2], [])) for img in images]

    def track(self, source: np.ndarray, **kwargs: object) -> list:
        self.calls.append(("track", [source.shape[:2]]))
        return []


def _tiled_detector(monkeypatch: pytest.MonkeyPatch, tiles: list | None) -> YoloByteTrackDetector:
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=_FakeYolo))
    settings = {"path": "m.engine", "backend": "tensorrt", "input_size": 640, "confidence": 0.35}
    detector = create_detector({**settings, "tiles": tiles})
    # Real ByteTrack needs ultralytics; number the merged detections instead.
    monkeypatch.setattr(
        detector,
        "_track",
        lambda img, dets: [Det(d.bbox, d.conf, i, d.keypoints) for i, d in enumerate(dets)],
    )
    return detector


def test_null_tiles_keep_the_single_full_frame_track_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    for tiles in (None, []):
        detector = _tiled_detector(monkeypatch, tiles)
        assert detector.infer(np.zeros((100, 200, 3), dtype=np.uint8)) == []
        assert detector._model.calls == [("track", [(100, 200)])]


def test_tiles_run_in_one_batch_map_back_and_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    # Full frame at the model conf (0.35) + two overlapping far-half tiles at conf 0.2.
    tiles = [[0, 0, 1, 1], [0, 0.5, 0.6, 1, 0.2], [0.35, 0.5, 1, 1, 0.2]]
    detector = _tiled_detector(monkeypatch, tiles)
    detector._model.answers = {
        # Full frame: near player, and the far player below the full-frame conf.
        (100, 200): [[10, 60, 30, 95, 0.8], [90, 55, 100, 75, 0.3]],
        # Left tile (x0 0, y0 50): near player again, the far player at 0.25, and a
        # person cut by the tile's inner right edge (dropped; the full frame sees them).
        (50, 120): [[10, 10, 30, 45, 0.6], [90, 5, 100, 25, 0.25], [105, 0, 120, 40, 0.9]],
        # Right tile (x0 70, y0 50): the far player again, shifted a pixel.
        (50, 130): [[21, 5, 31, 25, 0.22]],
    }
    players = detector.infer(np.zeros((100, 200, 3), dtype=np.uint8))
    assert detector._model.calls == [("predict", [(100, 200), (50, 120), (50, 130)])]
    assert [(p.bbox, p.conf) for p in players] == [
        ([10, 60, 30, 95], 0.8),
        ([90, 55, 100, 75], 0.25),
    ]
    assert players[1].keypoints == [[90, 75, 0.9]]


def test_bad_tile_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="tile"):
        _tiled_detector(monkeypatch, [[0.5, 0, 0.2, 1]])
