from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
benchmark = importlib.import_module("benchmark")
soak_test = importlib.import_module("soak_test")


def test_percentile_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]
    assert benchmark.percentile(values, 50) == 50.0
    assert benchmark.percentile(values, 99) == 99.0
    assert benchmark.percentile([7.0], 99) == 7.0
    assert benchmark.percentile([], 50) is None


def test_g1_thresholds() -> None:
    assert benchmark.g1_failures(60.0, 150.0, 250.0) == []
    assert len(benchmark.g1_failures(59.9, 150.1, 250.1)) == 3
    assert len(benchmark.g1_failures(None, None, None)) == 3


def test_build_config_adds_looping_replay_and_capture(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    override.write_text("server:\n  ws_port: 9999\n", encoding="utf-8")
    config = benchmark.build_config(tmp_path / "clip.mp4", override, loop=True)
    assert config["source"]["path"] == str(tmp_path / "clip.mp4")
    assert config["source"]["loop"] is True
    assert config["server"] == {"ws_port": 9999, "http_port": 8080}
    assert config["workers"]["capture"]["command"][1:] == ["-m", "hud.sources.capture"]


def test_memory_growth_compares_first_and_last_window() -> None:
    flat = [(float(t), 100.0) for t in range(0, 10800, 60)]
    assert soak_test.memory_growth(flat, 3600) == pytest.approx(0.0)
    growing = [(t, 100.0 if t <= 3600 else 110.0) for t, _ in flat]
    assert soak_test.memory_growth(growing, 3600) == pytest.approx(0.10)
    assert soak_test.memory_growth([(0.0, 1.0)], 3600) is None


def test_restart_failures() -> None:
    restarts = [
        {"worker": "players", "restart_s": 0.4},
        {"worker": "shuttle", "restart_s": 3.2},
        {"worker": "capture", "restart_s": None},
    ]
    assert [r["worker"] for r in soak_test.restart_failures(restarts)] == ["shuttle", "capture"]
