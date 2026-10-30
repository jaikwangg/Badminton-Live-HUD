from __future__ import annotations

import sys
import threading
import time

from hud.config import AppConfig, SupervisorConfig, load_config
from hud.supervisor import Supervisor


def test_default_config_has_expected_ring_and_timeout() -> None:
    config = load_config()
    assert config.bus.slots == 4
    assert config.supervisor.heartbeat_timeout_s == 3


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
