"""Restart configured worker processes when their heartbeat stream goes quiet."""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import TextIO

from hud.config import AppConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class WorkerProcess:
    command: list[str]
    process: subprocess.Popen[str] | None  # None: spawn failed, worker is down
    last_heartbeat: float
    reader: threading.Thread | None


class Supervisor:
    """Supervise configured commands; workers publish heartbeat JSON on stdout."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self._heartbeats: queue.Queue[tuple[str, float]] = queue.Queue(maxsize=256)
        self._workers: dict[str, WorkerProcess] = {}
        self._stopping = threading.Event()

    def _read_heartbeats(self, name: str, stream: TextIO) -> None:
        for line in stream:
            try:
                message = json.loads(line)
                if message.get("type") == "heartbeat" and message.get("worker") == name:
                    try:
                        self._heartbeats.put_nowait((name, time.monotonic()))
                    except queue.Full:
                        pass
            except (json.JSONDecodeError, AttributeError):
                continue

    def _start(self, name: str, command: list[str]) -> None:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except (OSError, ValueError):
            logger.exception("worker %s failed to start: %s", name, command)
            # Retry after one heartbeat timeout, like a worker that went quiet.
            self._workers[name] = WorkerProcess(command, None, time.monotonic(), None)
            return
        assert process.stdout is not None
        reader = threading.Thread(
            target=self._read_heartbeats, args=(name, process.stdout), daemon=True
        )
        reader.start()
        # The heartbeat timeout only starts counting after the startup grace period.
        first_deadline = time.monotonic() + self.config.supervisor.startup_grace_s
        self._workers[name] = WorkerProcess(command, process, first_deadline, reader)

    def start(self) -> None:
        for name, settings in self.config.workers.items():
            command = settings.get("command")
            if settings.get("enabled", False) and command:
                self._start(name, list(command))

    def run(self) -> None:
        self.start()
        timeout = self.config.supervisor.heartbeat_timeout_s
        while not self._stopping.wait(self.config.supervisor.poll_interval_s):
            while True:
                try:
                    name, received = self._heartbeats.get_nowait()
                except queue.Empty:
                    break
                if name in self._workers:
                    self._workers[name].last_heartbeat = received
            now = time.monotonic()
            for name, worker in list(self._workers.items()):
                running = worker.process is None or worker.process.poll() is None
                if running and now - worker.last_heartbeat <= timeout:
                    continue
                self._stop_worker(worker)
                self._start(name, worker.command)

    def stop(self) -> None:
        self._stopping.set()
        for worker in self._workers.values():
            self._stop_worker(worker)

    @staticmethod
    def _stop_worker(worker: WorkerProcess) -> None:
        if worker.process is None:
            return
        if worker.process.poll() is None:
            worker.process.terminate()
            try:
                worker.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait()
        if worker.process.stdout:
            worker.process.stdout.close()


def main() -> None:
    supervisor = Supervisor()
    try:
        supervisor.run()
    except KeyboardInterrupt:
        supervisor.stop()


if __name__ == "__main__":
    main()
