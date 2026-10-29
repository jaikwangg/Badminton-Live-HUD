"""Versioned JSON messages shared by workers and fusion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, TypeVar

T = TypeVar("T", bound="Message")


@dataclass(slots=True)
class Message:
    v: int = 1
    type: ClassVar[str]

    def to_json(self) -> str:
        return json.dumps({"v": self.v, "type": self.type, **asdict(self)}, separators=(",", ":"))

    @classmethod
    def from_json(cls: type[T], payload: str | bytes) -> T:
        values = json.loads(payload)
        if values.pop("type", None) != cls.type:
            raise ValueError(f"Expected message type {cls.type!r}")
        return cls(**values)


@dataclass(slots=True)
class PlayerDetection:
    track_id: int
    bbox: list[float]
    foot_px: list[float]
    conf: float


@dataclass(slots=True)
class PlayersMessage(Message):
    type: ClassVar[str] = "players"
    frame_id: int = 0
    ts: int = 0
    players: list[PlayerDetection] | list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: str | bytes) -> PlayersMessage:
        values = json.loads(payload)
        if values.pop("type", None) != cls.type:
            raise ValueError(f"Expected message type {cls.type!r}")
        if values.get("players") is not None:
            values["players"] = [
                item if isinstance(item, PlayerDetection) else PlayerDetection(**item)
                for item in values["players"]
            ]
        return cls(**values)


@dataclass(slots=True)
class ShuttleMessage(Message):
    type: ClassVar[str] = "shuttle"
    frame_id: int = 0
    ts: int = 0
    visible: bool = False
    px: list[float] | None = None
    conf: float | None = None


@dataclass(slots=True)
class HitMessage(Message):
    type: ClassVar[str] = "hit"
    ts: int = 0
    strength: float | None = None


@dataclass(slots=True)
class HeartbeatMessage(Message):
    type: ClassVar[str] = "heartbeat"
    worker: str = ""
    ts: int = 0
    fps: float | None = None
    proc_ms: float | None = None


@dataclass(slots=True)
class Frame:
    """Captured BGR image and its monotonic capture timestamp."""

    frame_id: int
    ts: int
    width: int
    height: int
    image: bytes


MESSAGE_TYPES: dict[str, type[Message]] = {
    cls.type: cls for cls in (PlayersMessage, ShuttleMessage, HitMessage, HeartbeatMessage)
}


def message_from_json(payload: str | bytes) -> Message:
    values = json.loads(payload)
    try:
        message_type = MESSAGE_TYPES[values["type"]]
    except (KeyError, TypeError) as exc:
        raise ValueError("Unknown message type") from exc
    return message_type.from_json(payload)
