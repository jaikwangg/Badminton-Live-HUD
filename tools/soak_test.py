"""Loop a clip for hours, kill random workers, and check G2 memory and G3 restart.

Needs psutil (process tree RSS). VRAM comes from nvidia-smi when present.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import STARTUP_TIMEOUT_S, System, build_config, get_health, nvidia_smi  # noqa: E402

# Thresholds from docs/GOALS.md.
G2_MAX_MEMORY_GROWTH = 0.05
G2_BASELINE_WINDOW_S = 3600.0  # first hour vs last hour
G3_MAX_RESTART_S = 3.0
RESTART_WAIT_S = 10.0
# Random kills hit model workers; server/recorder only with --kill-server-recorder.
MODEL_WORKERS = ("players", "shuttle", "audio")
SERVICE_WORKERS = ("server", "recorder")


def memory_growth(samples: list[tuple[float, float]], window_s: float) -> float | None:
    """Mean of the last `window_s` over the mean of the first `window_s`, minus 1."""
    if len(samples) < 2:
        return None
    first_t, last_t = samples[0][0], samples[-1][0]
    baseline = [value for t, value in samples if t <= first_t + window_s]
    end = [value for t, value in samples if t >= last_t - window_s]
    base = sum(baseline) / len(baseline)
    return (sum(end) / len(end)) / base - 1 if base > 0 else None


def restart_failures(restarts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Kills not followed by a respawn within G3_MAX_RESTART_S (None = never respawned)."""
    return [r for r in restarts if r["restart_s"] is None or r["restart_s"] > G3_MAX_RESTART_S]


def _worker_process(supervisor_pid: int, command: list[str]) -> Any:
    import psutil

    for child in psutil.Process(supervisor_pid).children():
        try:
            if child.cmdline()[1:] == list(command[1:]):
                return child
        except psutil.Error:
            continue
    return None


def kill_and_time_restart(system: System, name: str, command: list[str]) -> dict[str, Any]:
    """Kill one worker and measure until the supervisor has a new process running for it."""
    victim = _worker_process(system.supervisor_pid, command)
    if victim is None:
        return {"worker": name, "killed_pid": None, "restart_s": None}
    victim.kill()
    killed_at = time.monotonic()
    while (elapsed := time.monotonic() - killed_at) < RESTART_WAIT_S:
        replacement = _worker_process(system.supervisor_pid, command)
        if replacement is not None and replacement.pid != victim.pid:
            return {"worker": name, "killed_pid": victim.pid, "restart_s": elapsed}
        time.sleep(0.05)
    return {"worker": name, "killed_pid": victim.pid, "restart_s": None}


def sample_memory(system: System) -> tuple[float, float | None]:
    """(RSS MiB of every system process, VRAM MiB of those processes or whole GPU)."""
    import psutil

    pids = system.pids()
    rss = 0
    for pid in pids:
        try:
            rss += psutil.Process(pid).memory_info().rss
        except psutil.Error:
            continue
    vram = None
    apps = nvidia_smi("--query-compute-apps=pid,used_memory") or []
    used = [float(m) for p, m in apps if p.isdigit() and int(p) in pids and m.isdigit()]
    if used:
        vram = sum(used)
    else:
        # Windows WDDM reports per-process memory as N/A; fall back to the whole GPU.
        gpu = nvidia_smi("--query-gpu=memory.used")
        if gpu and gpu[0][0].replace(".", "", 1).isdigit():
            vram = float(gpu[0][0])
    return rss / 2**20, vram


def write_chart(rows: list[dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    hours = [row["elapsed_s"] / 3600 for row in rows]
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(hours, [row["rss_mib"] for row in rows], label="RSS (all processes)")
    vram = [(h, row["vram_mib"]) for h, row in zip(hours, rows, strict=True) if row["vram_mib"]]
    if vram:
        axis.plot([h for h, _ in vram], [v for _, v in vram], label="VRAM")
    axis.set_xlabel("hours")
    axis.set_ylabel("MiB")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path, help="Clip to loop")
    parser.add_argument("--config", type=Path, help="YAML merged over config/default.yaml")
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--kill-every-min", type=float, default=15.0, help="0 disables kills")
    parser.add_argument("--sample-every-s", type=float, default=60.0)
    parser.add_argument("--out-dir", type=Path, default=Path("soak"))
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--kill-server-recorder",
        action="store_true",
        help="Also pick the server and recorder for random kills",
    )
    args = parser.parse_args()
    if not args.replay.is_file():
        parser.error(f"clip not found: {args.replay}")
    import psutil  # noqa: F401  # fail early, not after hours

    rng = random.Random(args.seed)
    config = build_config(args.replay.resolve(), args.config, loop=True)
    targets = MODEL_WORKERS + (SERVICE_WORKERS if args.kill_server_recorder else ())
    workers = {
        n: w["command"] for n, w in config["workers"].items() if w.get("enabled") and n in targets
    }
    if args.kill_every_min > 0 and not workers:
        parser.error(f"no enabled worker to kill among {targets}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    duration_s = args.hours * 3600
    rows: list[dict[str, Any]] = []
    restarts: list[dict[str, Any]] = []
    crashed = False

    system = System(config)
    system.start()
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while not (get_health(config["server"]["http_port"]) or {}).get("fps"):
            if time.monotonic() > deadline or not system.alive():
                raise SystemExit("system did not start delivering frames")
            time.sleep(1)
        start = time.monotonic()
        next_sample = start
        next_kill = start + args.kill_every_min * 60 if args.kill_every_min > 0 else None
        with (args.out_dir / "soak.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, ["elapsed_s", "rss_mib", "vram_mib", "processes", "fps"]
            )
            writer.writeheader()
            while (now := time.monotonic()) - start < duration_s:
                if not system.alive():
                    crashed = True
                    break
                if now >= next_sample:
                    rss, vram = sample_memory(system)
                    health = get_health(config["server"]["http_port"]) or {}
                    row = {
                        "elapsed_s": round(now - start, 1),
                        "rss_mib": round(rss, 1),
                        "vram_mib": vram,
                        "processes": len(system.pids()),
                        "fps": health.get("fps"),
                    }
                    rows.append(row)
                    writer.writerow(row)
                    stream.flush()
                    next_sample += args.sample_every_s
                if next_kill is not None and now >= next_kill:
                    name = rng.choice(sorted(workers))
                    restarts.append(kill_and_time_restart(system, name, workers[name]))
                    restarts[-1]["at_s"] = round(now - start, 1)
                    print(json.dumps(restarts[-1]), flush=True)
                    next_kill += args.kill_every_min * 60
                time.sleep(0.5)
    finally:
        system.stop()

    window_s = min(G2_BASELINE_WINDOW_S, duration_s / 3)
    rss_growth = memory_growth([(r["elapsed_s"], r["rss_mib"]) for r in rows], window_s)
    vram_samples = [(r["elapsed_s"], r["vram_mib"]) for r in rows if r["vram_mib"] is not None]
    vram_growth = memory_growth(vram_samples, window_s)
    g2_pass = (
        not crashed
        and rss_growth is not None
        and rss_growth <= G2_MAX_MEMORY_GROWTH
        and (vram_growth is None or vram_growth <= G2_MAX_MEMORY_GROWTH)
    )
    failed_restarts = restart_failures(restarts)
    result = {
        "hours_requested": args.hours,
        "hours_run": rows[-1]["elapsed_s"] / 3600 if rows else 0.0,
        "crashed": crashed,
        "baseline_window_s": window_s,
        "rss_growth": rss_growth,
        "vram_growth": vram_growth,  # null: no nvidia-smi, not measured
        "g2_pass": g2_pass,
        "restarts": restarts,
        "g3_pass": not failed_restarts,
        "g3_note": "restart_s = kill until supervisor spawned a new process for that worker",
        "chart": "memory.png" if write_chart(rows, args.out_dir / "memory.png") else None,
    }
    result["pass"] = result["g2_pass"] and result["g3_pass"]
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "restarts"}, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
