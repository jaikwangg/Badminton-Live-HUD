"""Start the configured processes and restart them, with backoff, when they exit or go quiet."""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

from hud.config import AppConfig, SupervisorConfig, load_config

logger = logging.getLogger(__name__)


def resolve_command(command: list[str]) -> list[str]:
    """Run ``python``/``python3`` entries with this interpreter, not whatever PATH finds."""
    if command and command[0] in ("python", "python3"):
        return [sys.executable, *command[1:]]
    return command


def next_restart_delay(previous: float | None, uptime_s: float, config: SupervisorConfig) -> float:
    """Exponential backoff that resets once a worker has stayed up long enough."""
    if previous is None or uptime_s >= config.restart_reset_after_s:
        return config.restart_backoff_initial_s
    return min(previous * 2, config.restart_backoff_max_s)


@dataclass
class WorkerProcess:
    command: list[str]
    heartbeat: bool  # False: restart only when the process exits (server, recorder)
    log: logging.Logger  # the worker's own stderr/stdout log file
    process: subprocess.Popen[str] | None = None  # None: down, waiting for next_start_at
    last_heartbeat: float = 0.0
    started_at: float = 0.0
    next_start_at: float = 0.0
    restart_delay: float | None = None
    readers: list[threading.Thread] = field(default_factory=list)


class Supervisor:
    """Supervise configured commands; workers publish heartbeat JSON on stdout."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self._heartbeats: queue.Queue[tuple[str, float]] = queue.Queue(maxsize=256)
        self._workers: dict[str, WorkerProcess] = {}
        self._stopping = threading.Event()

    def _worker_log(self, name: str) -> logging.Logger:
        settings = self.config.supervisor
        logs_dir = Path(settings.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            logs_dir / f"{name}.log",
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backups,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        # A standalone logger: not registered globally, so restarts and tests do not pile up.
        worker_log = logging.Logger(f"hud.worker.{name}")
        worker_log.addHandler(handler)
        return worker_log

    def _read_stdout(self, name: str, stream: TextIO, worker_log: logging.Logger) -> None:
        try:
            for line in stream:
                try:
                    message = json.loads(line)
                    if message.get("type") == "heartbeat" and message.get("worker") == name:
                        try:
                            self._heartbeats.put_nowait((name, time.monotonic()))
                        except queue.Full:
                            pass
                        continue
                except (json.JSONDecodeError, AttributeError):
                    pass
                worker_log.info("stdout: %s", line.rstrip())
        except (OSError, ValueError):
            pass  # pipe closed by _stop_worker

    @staticmethod
    def _read_stderr(stream: TextIO, worker_log: logging.Logger) -> None:
        try:
            for line in stream:
                worker_log.info("%s", line.rstrip())
        except (OSError, ValueError):
            pass

    def _start(self, name: str, worker: WorkerProcess) -> None:
        now = time.monotonic()
        try:
            process = subprocess.Popen(
                worker.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except (OSError, ValueError) as exc:
            worker.log.info("supervisor: failed to start %s: %s", worker.command, exc)
            worker.started_at = now
            self._schedule_restart(name, worker, f"start failed: {exc}", None, now)
            return
        assert process.stdout is not None and process.stderr is not None
        worker.readers = [
            threading.Thread(
                target=self._read_stdout, args=(name, process.stdout, worker.log), daemon=True
            ),
            threading.Thread(
                target=self._read_stderr, args=(process.stderr, worker.log), daemon=True
            ),
        ]
        for reader in worker.readers:
            reader.start()
        worker.process = process
        worker.started_at = now
        # The heartbeat timeout only starts counting after the startup grace period.
        worker.last_heartbeat = now + self.config.supervisor.startup_grace_s
        logger.info("started %s pid %d: %s", name, process.pid, worker.command)
        worker.log.info("supervisor: started pid %d: %s", process.pid, worker.command)

    def _schedule_restart(
        self, name: str, worker: WorkerProcess, reason: str, code: int | None, now: float
    ) -> None:
        uptime = now - worker.started_at
        worker.restart_delay = next_restart_delay(
            worker.restart_delay, uptime, self.config.supervisor
        )
        worker.process = None
        worker.next_start_at = now + worker.restart_delay
        message = "%s down (%s, exit code %s, up %.1f s); restarting in %.1f s"
        args = (name, reason, code, uptime, worker.restart_delay)
        logger.warning(message, *args)
        worker.log.info("supervisor: " + message, *args)

    def start(self) -> None:
        for name, settings in self.config.workers.items():
            command = settings.get("command")
            if settings.get("enabled", False) and command:
                worker = WorkerProcess(
                    command=resolve_command(list(command)),
                    heartbeat=bool(settings.get("heartbeat", True)),
                    log=self._worker_log(name),
                )
                self._start(name, worker)
                self._workers[name] = worker

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
                if worker.process is None:
                    if now >= worker.next_start_at:
                        self._start(name, worker)
                    continue
                if worker.process.poll() is not None:
                    reason = "process exit"
                elif worker.heartbeat and now - worker.last_heartbeat > timeout:
                    reason = f"no heartbeat for {now - worker.last_heartbeat:.1f} s"
                else:
                    continue
                code = self._stop_worker(worker)
                self._schedule_restart(name, worker, reason, code, now)

    def stop(self) -> None:
        self._stopping.set()
        for name, worker in self._workers.items():
            if worker.process is not None:
                code = self._stop_worker(worker)
                logger.info("stopped %s (exit code %s)", name, code)
            for handler in worker.log.handlers:
                handler.close()

    @staticmethod
    def _stop_worker(worker: WorkerProcess) -> int | None:
        process = worker.process
        if process is None:
            return None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        # Readers hit EOF once the process is gone, unless a grandchild still holds the pipe.
        for reader in worker.readers:
            reader.join(timeout=1)
        for stream in (process.stdout, process.stderr):
            if stream:
                stream.close()
        return process.returncode


def main() -> None:
    config = load_config()
    logs_dir = Path(config.supervisor.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                logs_dir / "supervisor.log",
                maxBytes=config.supervisor.log_max_bytes,
                backupCount=config.supervisor.log_backups,
                encoding="utf-8",
            ),
        ],
    )
    supervisor = Supervisor(config)
    try:
        supervisor.run()
    except KeyboardInterrupt:
        supervisor.stop()


if __name__ == "__main__":
    main()
