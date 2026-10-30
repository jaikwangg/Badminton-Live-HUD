from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_EXPORT_PATH = Path(__file__).parents[1] / "tools" / "export_trt.py"
_EXPORT_SPEC = importlib.util.spec_from_file_location("export_trt", _EXPORT_PATH)
assert _EXPORT_SPEC is not None and _EXPORT_SPEC.loader is not None
_EXPORT_MODULE = importlib.util.module_from_spec(_EXPORT_SPEC)
_EXPORT_SPEC.loader.exec_module(_EXPORT_MODULE)
export_onnx = _EXPORT_MODULE.export_onnx


def test_malformed_checkpoint_fails_before_writing_onnx(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "bad.pt"
    output = tmp_path / "shuttle.onnx"
    torch.save({"epoch": 1, "model": {}, "param_dict": {"seq_len": 8}}, checkpoint)

    with pytest.raises(ValueError, match="seq_len and bg_mode"):
        export_onnx(tmp_path / "no-official-repo", checkpoint, output)

    assert not output.exists()


def _sigmoid_onnx(path: Path, batch: int | str = 1) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    shape = [batch, 3, 4, 4]
    graph = helper.make_graph(
        [helper.make_node("Sigmoid", ["x"], ["y"])],
        "sigmoid",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))


def test_build_engine_uses_trtexec_when_on_path(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(_EXPORT_MODULE.shutil, "which", lambda name: "/opt/trtexec")
    monkeypatch.setattr(
        _EXPORT_MODULE.subprocess, "run", lambda command, check: calls.append(command)
    )

    _EXPORT_MODULE.build_engine(
        tmp_path / "m.onnx", tmp_path / "m.engine", fp16=True, workspace_mib=512
    )

    assert calls[0][0] == "/opt/trtexec"
    assert "--fp16" in calls[0] and "--memPoolSize=workspace:512M" in calls[0]


def test_build_engine_without_trtexec_or_tensorrt_fails_clearly(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(_EXPORT_MODULE.shutil, "which", lambda name: None)
    monkeypatch.setitem(sys.modules, "tensorrt", None)

    with pytest.raises(RuntimeError, match="trtexec on PATH or the TensorRT Python package"):
        _EXPORT_MODULE.build_engine(tmp_path / "m.onnx", tmp_path / "m.engine", fp16=False)


def _python_build_or_skip(monkeypatch, onnx_path: Path, engine_path: Path) -> None:
    pytest.importorskip("tensorrt")
    monkeypatch.setattr(_EXPORT_MODULE.shutil, "which", lambda name: None)
    try:
        _EXPORT_MODULE.build_engine(onnx_path, engine_path, fp16=True, workspace_mib=256)
    except TypeError as exc:  # pybind raises this when no CUDA device or driver is present
        pytest.skip(f"TensorRT builder unavailable: {exc}")


def test_python_api_builds_fp16_engine_without_trtexec(monkeypatch, tmp_path: Path) -> None:
    onnx_path, engine_path = tmp_path / "m.onnx", tmp_path / "out" / "m.engine"
    _sigmoid_onnx(onnx_path)

    _python_build_or_skip(monkeypatch, onnx_path, engine_path)

    import tensorrt as trt

    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
        engine_path.read_bytes()
    )
    assert engine is not None
    assert tuple(engine.get_tensor_shape("x")) == (1, 3, 4, 4)


def test_python_api_rejects_dynamic_onnx_shapes(monkeypatch, tmp_path: Path) -> None:
    onnx_path = tmp_path / "m.onnx"
    _sigmoid_onnx(onnx_path, batch="N")

    with pytest.raises(ValueError, match="dynamic shape"):
        _python_build_or_skip(monkeypatch, onnx_path, tmp_path / "m.engine")
    assert not (tmp_path / "m.engine").exists()
