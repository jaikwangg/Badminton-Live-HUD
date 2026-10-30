from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from hud.config import AppConfig, SupervisorConfig, load_config
from hud.supervisor import Supervisor


def test_default_config_has_expected_ring_and_timeout() -> None:
    config = load_config()
    assert config.bus.slots == 4
    assert config.supervisor.heartbeat_timeout_s == 3


def test_default_config_resolves_from_cwd_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "default.yaml").write_text("bus:\n  slots: 8\n", encoding="utf-8")
    override = tmp_path / "override.yaml"
    override.write_text("bus:\n  slots: 6\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HUD_DEFAULT_CONFIG", raising=False)
    assert load_config().bus.slots == 8
    monkeypatch.setenv("HUD_DEFAULT_CONFIG", str(override))
    assert load_config().bus.slots == 6


def test_fusion_section_is_plain_dict_with_empty_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HUD_DEFAULT_CONFIG", raising=False)
    fusion = load_config().fusion
    assert fusion["identity"]["app_window_frames"] == 90
    assert fusion["rally"]["end_gap_ms"] == 1800
    assert fusion["shuttle"]["max_px_per_s"] is None
    bare = tmp_path / "bare.yaml"
    bare.write_text("bus:\n  slots: 4\n", encoding="utf-8")
    monkeypatch.setenv("HUD_DEFAULT_CONFIG", str(bare))
    assert load_config().fusion == {}


HEARTBEAT_COMMAND = [
    sys.executable,
    "-u",
    "-c",
    "import json, time\n"
    "while True:\n"
    " print(json.dumps({'type': 'heartbeat', 'worker': 'demo', 'v': 1}), flush=True)\n"
    " time.sleep(.05)\n",
]


def test_supervisor_keeps_worker_that_sends_heartbeats() -> None:
    config = AppConfig(
        workers={"demo": {"enabled": True, "command": HEARTBEAT_COMMAND}},
        supervisor=SupervisorConfig(heartbeat_timeout_s=0.5, poll_interval_s=0.05),
    )
    supervisor = Supervisor(config)
    runner = threading.Thread(target=supervisor.run, daemon=True)
    try:
        runner.start()
        deadline = time.monotonic() + 2
        while "demo" not in supervisor._workers and time.monotonic() < deadline:
            time.sleep(0.01)
        first_pid = supervisor._workers["demo"].process.pid
        time.sleep(1.5)  # three heartbeat timeouts
        assert supervisor._workers["demo"].process.pid == first_pid
    finally:
        supervisor.stop()
        runner.join(timeout=2)


def test_supervisor_survives_missing_executable() -> None:
    config = AppConfig(
        workers={"demo": {"enabled": True, "command": ["hud-no-such-executable-xyz"]}},
        supervisor=SupervisorConfig(heartbeat_timeout_s=0.1, poll_interval_s=0.05),
    )
    supervisor = Supervisor(config)
    runner = threading.Thread(target=supervisor.run, daemon=True)
    try:
        runner.start()
        deadline = time.monotonic() + 2
        while "demo" not in supervisor._workers and time.monotonic() < deadline:
            time.sleep(0.01)
        first_attempt = supervisor._workers["demo"].last_heartbeat
        time.sleep(0.5)  # several retry periods
        assert runner.is_alive()
        assert supervisor._workers["demo"].process is None
        assert supervisor._workers["demo"].last_heartbeat > first_attempt  # retried
    finally:
        supervisor.stop()
        runner.join(timeout=2)


def test_supervisor_keeps_slow_starting_worker_during_grace() -> None:
    silent_command = [sys.executable, "-c", "import time; time.sleep(30)"]
    config = AppConfig(
        workers={"demo": {"enabled": True, "command": silent_command}},
        supervisor=SupervisorConfig(
            heartbeat_timeout_s=0.2, poll_interval_s=0.05, startup_grace_s=2.0
        ),
    )
    supervisor = Supervisor(config)
    runner = threading.Thread(target=supervisor.run, daemon=True)
    try:
        runner.start()
        deadline = time.monotonic() + 2
        while "demo" not in supervisor._workers and time.monotonic() < deadline:
            time.sleep(0.01)
        first_pid = supervisor._workers["demo"].process.pid
        time.sleep(1.0)  # five heartbeat timeouts, still inside the grace period
        assert supervisor._workers["demo"].process.pid == first_pid
    finally:
        supervisor.stop()
        runner.join(timeout=2)


def test_supervisor_restarts_killed_worker() -> None:
    command = HEARTBEAT_COMMAND
    config = AppConfig(
        workers={"demo": {"enabled": True, "command": command}},
        supervisor=SupervisorConfig(heartbeat_timeout_s=3, poll_interval_s=0.05),
    )
    supervisor = Supervisor(config)
    runner = threading.Thread(target=supervisor.run, daemon=True)
    try:
        runner.start()
        deadline = time.monotonic() + 2
        while "demo" not in supervisor._workers and time.monotonic() < deadline:
            time.sleep(0.01)
        first_pid = supervisor._workers["demo"].process.pid
        supervisor._workers["demo"].process.kill()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            worker = supervisor._workers.get("demo")
            if worker is not None and worker.process.pid != first_pid:
                break
            time.sleep(0.02)
        assert supervisor._workers["demo"].process.pid != first_pid
    finally:
        supervisor.stop()
        runner.join(timeout=2)
