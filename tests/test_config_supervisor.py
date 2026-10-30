from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from hud import config as hud_config
from hud.config import AppConfig, BusConfig, SupervisorConfig, endpoint, load_config
from hud.supervisor import Supervisor, next_restart_delay, resolve_command


@pytest.fixture(autouse=True)
def _logs_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Supervisor writes logs/<worker>.log relative to cwd.
    monkeypatch.chdir(tmp_path)


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
        supervisor=SupervisorConfig(
            heartbeat_timeout_s=0.1, poll_interval_s=0.05, restart_backoff_initial_s=0.05
        ),
    )
    supervisor = Supervisor(config)
    runner = threading.Thread(target=supervisor.run, daemon=True)
    try:
        runner.start()
        deadline = time.monotonic() + 2
        while "demo" not in supervisor._workers and time.monotonic() < deadline:
            time.sleep(0.01)
        first_attempt = supervisor._workers["demo"].next_start_at
        time.sleep(0.5)  # several retry periods
        assert runner.is_alive()
        assert supervisor._workers["demo"].process is None
        assert supervisor._workers["demo"].next_start_at > first_attempt  # retried
        assert supervisor._workers["demo"].restart_delay > 0.05  # backed off
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
            if worker is not None and worker.process and worker.process.pid != first_pid:
                break
            time.sleep(0.02)
        assert supervisor._workers["demo"].process.pid != first_pid
    finally:
        supervisor.stop()
        runner.join(timeout=2)


def test_endpoint_prefers_config_then_platform_default(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AppConfig(bus=BusConfig(endpoints={"players": "tcp://10.0.0.2:6000", "audio": None}))
    assert endpoint(config, "players") == "tcp://10.0.0.2:6000"
    monkeypatch.setattr(hud_config.sys, "platform", "linux")
    assert endpoint(config, "shuttle") == "ipc:///tmp/hud_shuttle"
    assert endpoint(config, "audio") == "ipc:///tmp/hud_audio"
    monkeypatch.setattr(hud_config.sys, "platform", "win32")
    assert endpoint(config, "shuttle") == "tcp://127.0.0.1:5762"
    assert len({endpoint(config, name) for name in ("players", "shuttle", "audio")}) == 3


def test_default_config_endpoints_and_calibration_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HUD_DEFAULT_CONFIG", raising=False)
    config = load_config()
    assert config.bus.endpoints == {"players": None, "shuttle": None, "audio": None}
    assert config.models["players"]["roi"] is None
    repo_config = Path(hud_config.__file__).resolve().parents[2] / "config"
    assert Path(config.fusion["calibration_path"]) == repo_config / "court_calib.json"
    user = tmp_path / "site" / "user.yaml"
    user.parent.mkdir()
    user.write_text("fusion:\n  calibration_path: calib/court.json\n", encoding="utf-8")
    assert Path(load_config(user).fusion["calibration_path"]) == user.parent / "calib/court.json"
    absolute = tmp_path / "abs.json"
    user.write_text(f"fusion:\n  calibration_path: '{absolute}'\n", encoding="utf-8")
    assert Path(load_config(user).fusion["calibration_path"]) == absolute


def test_python_command_uses_this_interpreter() -> None:
    assert resolve_command(["python", "-m", "x"]) == [sys.executable, "-m", "x"]
    assert resolve_command(["python3", "-c", "1"]) == [sys.executable, "-c", "1"]
    assert resolve_command(["ffmpeg", "-i"]) == ["ffmpeg", "-i"]


def test_restart_backoff_doubles_caps_and_resets() -> None:
    settings = SupervisorConfig(
        restart_backoff_initial_s=0.5, restart_backoff_max_s=3.0, restart_reset_after_s=30.0
    )
    delays: list[float] = []
    delay = None
    for _ in range(5):
        delay = next_restart_delay(delay, 1.0, settings)
        delays.append(delay)
    assert delays == [0.5, 1.0, 2.0, 3.0, 3.0]
    assert next_restart_delay(3.0, 30.0, settings) == 0.5


def _wait_for(predicate: Callable[[], bool], seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def test_heartbeat_false_process_is_kept_while_running_and_logs_output(tmp_path: Path) -> None:
    command = [
        "python",
        "-c",
        "import sys, time\n"
        "print('hello stdout', flush=True)\n"
        "print('boom on stderr', file=sys.stderr, flush=True)\n"
        "time.sleep(30)\n",
    ]
    config = AppConfig(
        workers={"server": {"enabled": True, "heartbeat": False, "command": command}},
        supervisor=SupervisorConfig(
            heartbeat_timeout_s=0.1, poll_interval_s=0.05, startup_grace_s=0.0
        ),
    )
    supervisor = Supervisor(config)
    runner = threading.Thread(target=supervisor.run, daemon=True)
    log_file = tmp_path / "logs" / "server.log"
    try:
        runner.start()
        assert _wait_for(lambda: "server" in supervisor._workers)
        first_pid = supervisor._workers["server"].process.pid
        assert _wait_for(lambda: "boom on stderr" in _read(log_file))
        time.sleep(0.5)  # many heartbeat timeouts without a single heartbeat
        assert supervisor._workers["server"].process.pid == first_pid
        text = _read(log_file)
        assert "stdout: hello stdout" in text
        assert f"started pid {first_pid}" in text
    finally:
        supervisor.stop()
        runner.join(timeout=2)


def test_exit_is_logged_with_code_and_restarted_with_backoff(tmp_path: Path) -> None:
    config = AppConfig(
        workers={"demo": {"enabled": True, "command": ["python", "-c", "raise SystemExit(3)"]}},
        supervisor=SupervisorConfig(poll_interval_s=0.05, restart_backoff_initial_s=0.1),
    )
    supervisor = Supervisor(config)
    runner = threading.Thread(target=supervisor.run, daemon=True)
    log_file = tmp_path / "logs" / "demo.log"
    try:
        runner.start()
        assert _wait_for(lambda: _read(log_file).count("process exit, exit code 3") >= 2)
        assert "restarting in 0.2 s" in _read(log_file)
    finally:
        supervisor.stop()
        runner.join(timeout=2)
