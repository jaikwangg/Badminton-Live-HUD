from __future__ import annotations

import json

from hud.bus.messages import PlayersMessage
from hud.models.detector import Det
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
