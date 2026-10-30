from __future__ import annotations

import ctypes
import enum
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from hud.models.shuttle import TensorRTSession, _cuda_check


class _FakeCudart(types.ModuleType):
    """CUDA runtime stand-in: 'device' memory is host memory, so copies are plain memmoves."""

    class cudaMemcpyKind(enum.IntEnum):  # noqa: N801 - mirrors the cuda-bindings name
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2

    def __init__(self) -> None:
        super().__init__("cuda.bindings.runtime")
        self.allocations: dict[int, ctypes.Array[ctypes.c_char]] = {}
        self.copies: list[int] = []
        self.synchronised = 0

    def cudaStreamCreate(self) -> tuple[int, int]:  # noqa: N802
        return 0, 7

    def cudaMalloc(self, size: int) -> tuple[int, int]:  # noqa: N802
        block = ctypes.create_string_buffer(size)
        self.allocations[ctypes.addressof(block)] = block
        return 0, ctypes.addressof(block)

    def cudaFree(self, pointer: int) -> tuple[int]:  # noqa: N802
        del self.allocations[pointer]
        return (0,)

    def cudaMemcpyAsync(  # noqa: N802
        self, destination: int, source: int, size: int, kind: int, stream: int
    ) -> tuple[int]:
        assert stream == 7
        ctypes.memmove(destination, source, size)
        self.copies.append(int(kind))
        return (0,)

    def cudaStreamSynchronize(self, stream: int) -> tuple[int]:  # noqa: N802
        self.synchronised += 1
        return (0,)


class _FakeContext:
    def __init__(self) -> None:
        self.addresses: dict[str, int] = {}
        self.streams: list[int] = []

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return (1, 4)

    def set_tensor_address(self, name: str, address: int) -> None:
        self.addresses[name] = address

    def execute_async_v3(self, stream: int) -> bool:
        # "Kernel": output = input * 2, reading and writing the bound device addresses.
        self.streams.append(stream)
        source = (ctypes.c_float * 4).from_address(self.addresses["x"])
        target = (ctypes.c_float * 4).from_address(self.addresses["y"])
        for index in range(4):
            target[index] = source[index] * 2
        return True


class _FakeEngine:
    num_io_tensors = 2

    def __init__(self) -> None:
        self.context = _FakeContext()

    def get_tensor_name(self, index: int) -> str:
        return ["x", "y"][index]

    def get_tensor_mode(self, name: str) -> str:
        return "in" if name == "x" else "out"

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return (1, 4)

    def get_tensor_dtype(self, name: str) -> str:
        return "float"

    def create_execution_context(self) -> _FakeContext:
        return self.context


def _fake_tensorrt(engine: _FakeEngine) -> types.ModuleType:
    trt = types.ModuleType("tensorrt")

    class Logger:
        ERROR = 1

        def __init__(self, level: int) -> None:
            pass

    class Runtime:
        def __init__(self, logger: Logger) -> None:
            pass

        def deserialize_cuda_engine(self, data: bytes) -> _FakeEngine:
            assert data == b"engine"
            return engine

    trt.Logger = Logger  # type: ignore[attr-defined]
    trt.Runtime = Runtime  # type: ignore[attr-defined]
    trt.TensorIOMode = types.SimpleNamespace(INPUT="in")  # type: ignore[attr-defined]
    trt.nptype = lambda dtype: np.float32  # type: ignore[attr-defined]
    return trt


def test_tensorrt_session_binds_addresses_and_copies_through_cuda_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _FakeEngine()
    cudart = _FakeCudart()
    bindings = types.ModuleType("cuda.bindings")
    bindings.runtime = cudart  # type: ignore[attr-defined]
    cuda = types.ModuleType("cuda")
    cuda.bindings = bindings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt(engine))
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", cudart)
    path = tmp_path / "model.engine"
    path.write_bytes(b"engine")

    session = TensorRTSession(path)
    assert [info.name for info in session.get_inputs()] == ["x"]
    assert [info.name for info in session.get_outputs()] == ["y"]
    for values in ([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]):
        (result,) = session.run(None, {"x": np.array([values], dtype=np.float64)})
        np.testing.assert_array_equal(result, np.array([values], dtype=np.float32) * 2)

    assert engine.context.streams == [7, 7]
    assert cudart.copies == [1, 2, 1, 2]
    assert cudart.synchronised == 2
    assert len(cudart.allocations) == 2  # buffers reused across runs, not reallocated


def test_cuda_check_unpacks_values_and_raises_on_error() -> None:
    assert _cuda_check((0, 42)) == 42
    with pytest.raises(RuntimeError, match="CUDA runtime call failed"):
        _cuda_check((2, 0))


def test_real_tensorrt_engine_matches_numpy(tmp_path: Path) -> None:
    trt = pytest.importorskip("tensorrt")
    pytest.importorskip("cuda.bindings.runtime")
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Sigmoid", ["x"], ["y"])],
        "sigmoid",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    logger = trt.Logger(trt.Logger.ERROR)
    try:
        builder = trt.Builder(logger)
    except (RuntimeError, TypeError) as exc:  # no CUDA device or driver
        pytest.skip(f"TensorRT builder unavailable: {exc}")
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    assert parser.parse(model.SerializeToString())
    serialized = builder.build_serialized_network(network, builder.create_builder_config())
    assert serialized is not None
    path = tmp_path / "sigmoid.engine"
    path.write_bytes(bytes(serialized))

    session = TensorRTSession(path)
    value = np.linspace(-3, 3, 48, dtype=np.float32).reshape(1, 3, 4, 4)
    (result,) = session.run(None, {"x": value})
    np.testing.assert_allclose(result, 1 / (1 + np.exp(-value)), atol=1e-5)
