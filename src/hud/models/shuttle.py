"""TrackNetV3 inference interface with ONNX Runtime and TensorRT engine backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hud.bus.messages import Frame


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    shape: tuple[int | str | None, ...]


class Session(Protocol):
    def get_inputs(self) -> list[TensorInfo]: ...

    def get_outputs(self) -> list[TensorInfo]: ...

    def run(
        self, output_names: list[str] | None, values: dict[str, np.ndarray]
    ) -> list[np.ndarray]: ...


class ShuttleTracker(Protocol):
    def infer(self, frames: Sequence[Frame]) -> tuple[bool, list[float] | None, float | None]:
        """Return visible, source-image [x,y], and confidence from three frames."""


class TensorRTSession:
    """Small TensorRT 10+ adapter for a serialized engine and one CUDA context."""

    def __init__(self, path: str | Path) -> None:
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "Install TensorRT and PyCUDA to load a .engine shuttle model"
            ) from exc
        self._cuda = cuda
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(self._logger)
        self._engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self._engine is None or not hasattr(self._engine, "num_io_tensors"):
            raise RuntimeError("TensorRT 10 or newer and a valid serialized engine are required")
        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()
        self._names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        self._inputs = [
            name
            for name in self._names
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self._outputs = [name for name in self._names if name not in self._inputs]
        self._buffers: dict[str, tuple[tuple[int, ...], np.dtype[Any], Any]] = {}

    def get_inputs(self) -> list[TensorInfo]:
        return [
            TensorInfo(name, tuple(self._engine.get_tensor_shape(name))) for name in self._inputs
        ]

    def get_outputs(self) -> list[TensorInfo]:
        return [
            TensorInfo(name, tuple(self._engine.get_tensor_shape(name))) for name in self._outputs
        ]

    def _buffer(self, name: str, shape: tuple[int, ...], dtype: np.dtype[Any]) -> Any:
        current = self._buffers.get(name)
        if current is not None and current[0] == shape and current[1] == dtype:
            return current[2]
        if current is not None:
            current[2].free()
        allocation = self._cuda.mem_alloc(int(np.prod(shape)) * dtype.itemsize)
        self._buffers[name] = (shape, dtype, allocation)
        return allocation

    def run(
        self, output_names: list[str] | None, values: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        for name, value in values.items():
            self._context.set_input_shape(name, tuple(value.shape))
        host_inputs: dict[str, np.ndarray] = {}
        for name, value in values.items():
            dtype = self._trt.nptype(self._engine.get_tensor_dtype(name))
            host = np.ascontiguousarray(value, dtype=dtype)
            host_inputs[name] = host
            buffer = self._buffer(name, tuple(host.shape), host.dtype)
            self._cuda.memcpy_htod_async(buffer, host, self._stream)
            self._context.set_tensor_address(name, int(buffer))

        output_values: list[np.ndarray] = []
        selected = output_names or self._outputs
        output_buffers: list[tuple[np.ndarray, Any]] = []
        for name in selected:
            shape = tuple(int(dim) for dim in self._context.get_tensor_shape(name))
            dtype = np.dtype(self._trt.nptype(self._engine.get_tensor_dtype(name)))
            host = np.empty(shape, dtype=dtype)
            buffer = self._buffer(name, shape, dtype)
            self._context.set_tensor_address(name, int(buffer))
            output_buffers.append((host, buffer))
        if not self._context.execute_async_v3(self._stream.handle):
            raise RuntimeError("TensorRT shuttle inference failed")
        for host, buffer in output_buffers:
            self._cuda.memcpy_dtoh_async(host, buffer, self._stream)
            output_values.append(host)
        self._stream.synchronize()
        return output_values


class TrackNetV3Tracker:
    """Preprocess three BGR frames and decode a TrackNet heat map."""

    def __init__(
        self,
        path: str | Path,
        backend: str,
        *,
        width: int = 512,
        height: int = 288,
        confidence: float = 0.5,
        session: Session | None = None,
    ) -> None:
        self.path = Path(path)
        self.backend = backend
        self.width = width
        self.height = height
        self.confidence = confidence
        if session is not None:
            self.session = session
        elif backend == "onnxruntime":
            if self.path.suffix.lower() != ".onnx":
                raise ValueError("ONNX Runtime shuttle backend requires a .onnx model")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("Install ONNX Runtime to load the shuttle model") from exc
            self.session = ort.InferenceSession(
                str(self.path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        elif backend == "tensorrt":
            if self.path.suffix.lower() != ".engine":
                raise ValueError("TensorRT shuttle backend requires a .engine model")
            self.session = TensorRTSession(self.path)
        else:
            raise ValueError(f"Unsupported shuttle backend: {backend}")
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("TrackNetV3 engine must have exactly one input tensor")
        self.input_name = inputs[0].name
        self.input_shape = inputs[0].shape
        outputs = self.session.get_outputs()
        if not outputs:
            raise ValueError("TrackNetV3 engine must provide an output heat map")
        self.output_name = outputs[0].name

    def _frame_array(self, frame: Frame) -> np.ndarray:
        image = np.frombuffer(frame.image, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        rows = np.linspace(0, frame.height - 1, self.height).astype(np.intp)
        columns = np.linspace(0, frame.width - 1, self.width).astype(np.intp)
        return image[rows[:, None], columns[None, :], :].astype(np.float32) / 255.0

    def _input_tensor(self, frames: Sequence[Frame]) -> np.ndarray:
        if len(frames) != 3:
            raise ValueError("TrackNetV3 requires exactly three consecutive frames")
        images = [self._frame_array(frame) for frame in frames]
        shape = self.input_shape
        if len(shape) == 4 and shape[1] == 9:
            return np.concatenate([image.transpose(2, 0, 1) for image in images], axis=0)[None, ...]
        if len(shape) == 4 and shape[-1] == 9:
            return np.concatenate(images, axis=2)[None, ...]
        if len(shape) == 5 and shape[1] == 3 and shape[-1] == 3:
            return np.stack(images, axis=0)[None, ...]
        raise ValueError(f"Unsupported TrackNetV3 input tensor shape: {shape}")

    def infer(self, frames: Sequence[Frame]) -> tuple[bool, list[float] | None, float | None]:
        values = {self.input_name: self._input_tensor(frames)}
        output = self.session.run([self.output_name], values)[0]
        heatmap = np.asarray(output, dtype=np.float32).squeeze()
        if heatmap.ndim != 2 or heatmap.size == 0:
            raise ValueError(f"Unsupported TrackNetV3 heat map shape: {output.shape}")
        finite = heatmap[np.isfinite(heatmap)]
        if finite.size == 0:
            return False, None, None
        if float(finite.min()) < 0 or float(finite.max()) > 1:
            heatmap = 1.0 / (1.0 + np.exp(-np.clip(heatmap, -40, 40)))
        heatmap = np.nan_to_num(heatmap, nan=-np.inf)
        flat_index = int(np.nanargmax(heatmap))
        confidence = float(heatmap.flat[flat_index])
        if confidence < self.confidence:
            return False, None, confidence
        y, x = np.unravel_index(flat_index, heatmap.shape)
        frame = frames[-1]
        px = [float(x) * frame.width / heatmap.shape[1], float(y) * frame.height / heatmap.shape[0]]
        return True, px, confidence


def create_shuttle_tracker(settings: dict[str, Any]) -> ShuttleTracker:
    return TrackNetV3Tracker(
        path=str(settings["path"]),
        backend=str(settings["backend"]),
        width=int(settings["width"]),
        height=int(settings["height"]),
        confidence=float(settings["confidence"]),
    )
