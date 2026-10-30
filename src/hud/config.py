"""Typed configuration loading with defaults from config/default.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SourceConfig:
    type: str = "file"
    path: str | None = None
    width: int = 1280
    height: int = 720
    fps: float = 60.0
    loop: bool = False


@dataclass(slots=True)
class BusConfig:
    name: str = "hud_frames"
    slots: int = 4


@dataclass(slots=True)
class SupervisorConfig:
    heartbeat_timeout_s: float = 3.0
    poll_interval_s: float = 0.25


@dataclass(slots=True)
class AppConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    bus: BusConfig = field(default_factory=BusConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    workers: dict[str, dict[str, Any]] = field(default_factory=dict)
    server: dict[str, Any] = field(default_factory=dict)
    recording: dict[str, Any] = field(default_factory=dict)


def _merge_defaults(values: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(value, merged[key])
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML config, filling omitted settings from the repository defaults."""
    default_path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    with default_path.open(encoding="utf-8") as stream:
        defaults = yaml.safe_load(stream) or {}
    user_values: dict[str, Any] = {}
    if path is not None:
        with Path(path).open(encoding="utf-8") as stream:
            user_values = yaml.safe_load(stream) or {}
    values = _merge_defaults(user_values, defaults)
    return AppConfig(
        source=SourceConfig(**values["source"]),
        bus=BusConfig(**values["bus"]),
        supervisor=SupervisorConfig(**values["supervisor"]),
        models=values.get("models", {}),
        workers=values.get("workers", {}),
        server=values.get("server", {}),
        recording=values.get("recording", {}),
    )
