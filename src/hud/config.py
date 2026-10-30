"""Typed configuration loading with defaults from config/default.yaml."""

from __future__ import annotations

import os
import sys
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
    # Worker PUB endpoints by worker name; a missing or null entry uses the platform default.
    endpoints: dict[str, str | None] = field(default_factory=dict)


@dataclass(slots=True)
class SupervisorConfig:
    heartbeat_timeout_s: float = 3.0
    poll_interval_s: float = 0.25
    startup_grace_s: float = 30.0
    restart_backoff_initial_s: float = 0.5
    restart_backoff_max_s: float = 10.0
    restart_reset_after_s: float = 30.0
    logs_dir: str = "logs"
    log_max_bytes: int = 10_000_000
    log_backups: int = 3


@dataclass(slots=True)
class AppConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    bus: BusConfig = field(default_factory=BusConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    workers: dict[str, dict[str, Any]] = field(default_factory=dict)
    fusion: dict[str, Any] = field(default_factory=dict)
    server: dict[str, Any] = field(default_factory=dict)
    recording: dict[str, Any] = field(default_factory=dict)


# pyzmq on Windows has no ipc:// transport, so workers fall back to fixed loopback ports.
_WINDOWS_PORTS = {"players": 5761, "shuttle": 5762, "audio": 5763}


def endpoint(config: AppConfig, name: str) -> str:
    """ZeroMQ endpoint of worker ``name``: ``bus.endpoints[name]`` or the platform default."""
    configured = config.bus.endpoints.get(name)
    if configured:
        return configured
    if sys.platform == "win32":
        return f"tcp://127.0.0.1:{_WINDOWS_PORTS[name]}"
    return f"ipc:///tmp/hud_{name}"


def _resolve_paths(values: dict[str, Any], base_dir: Path) -> None:
    """Resolve file paths in one config file against that file's directory."""
    fusion = values.get("fusion")
    if isinstance(fusion, dict) and fusion.get("calibration_path"):
        fusion["calibration_path"] = str(base_dir / fusion["calibration_path"])


def _merge_defaults(values: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(value, merged[key])
        else:
            merged[key] = value
    return merged


def _default_config_path() -> Path | None:
    """HUD_DEFAULT_CONFIG wins, then ./config/default.yaml, then the source checkout."""
    override = os.environ.get("HUD_DEFAULT_CONFIG")
    if override:
        return Path(override)
    for candidate in (
        Path.cwd() / "config" / "default.yaml",
        Path(__file__).resolve().parents[2] / "config" / "default.yaml",
    ):
        if candidate.is_file():
            return candidate
    return None


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML config, filling omitted settings from the default config file."""
    defaults: dict[str, Any] = {}
    default_path = _default_config_path()
    if default_path is not None:
        with default_path.open(encoding="utf-8") as stream:
            defaults = yaml.safe_load(stream) or {}
        _resolve_paths(defaults, default_path.resolve().parent)
    user_values: dict[str, Any] = {}
    if path is not None:
        with Path(path).open(encoding="utf-8") as stream:
            user_values = yaml.safe_load(stream) or {}
        _resolve_paths(user_values, Path(path).resolve().parent)
    values = _merge_defaults(user_values, defaults)
    return AppConfig(
        source=SourceConfig(**values.get("source", {})),
        bus=BusConfig(**values.get("bus", {})),
        supervisor=SupervisorConfig(**values.get("supervisor", {})),
        models=values.get("models", {}),
        workers=values.get("workers", {}),
        fusion=values.get("fusion", {}),
        server=values.get("server", {}),
        recording=values.get("recording", {}),
    )
