from __future__ import annotations

import importlib.util
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
