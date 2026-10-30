"""TrackNetV3 inference interface for ONNX Runtime and TensorRT backends."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    shape: tuple[int | str | None, ...]


@dataclass(frozen=True, slots=True)
class ResizedFrame:
    """An RGB frame resized once for the TrackNet input, with source geometry retained.

    source_width/height are the size of the region the model saw (the crop, or the full
    frame); origin_x/y place that region's top-left corner in full-frame pixels.
    """

    frame_id: int
    ts: int
    source_width: int
    source_height: int
    image_rgb: np.ndarray
    origin_x: int = 0
    origin_y: int = 0
    # Normalised float32 CHW copy, built once so each inference only copies it.
    chw: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chw", _normalised_chw(self.image_rgb))


def _normalised_chw(image_rgb: np.ndarray) -> np.ndarray:
    return image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0


class Session(Protocol):
    def get_inputs(self) -> list[TensorInfo]: ...

    def get_outputs(self) -> list[TensorInfo]: ...

    def run(
        self, output_names: list[str] | None, values: dict[str, np.ndarray]
    ) -> list[np.ndarray]: ...


class ShuttleTracker(Protocol):
    seq_len: int
    width: int
    height: int

    def warmup(self) -> None: ...

    def infer(
        self, frames: Sequence[ResizedFrame], background: np.ndarray
    ) -> tuple[bool, list[float] | None, float | None]: ...


def _cuda_check(result: tuple[Any, ...]) -> Any:
    """Unpack a cuda-bindings ``(error, *values)`` tuple, raising on any CUDA error."""
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"CUDA runtime call failed: {error!r}")
    return values[0] if values else None


class TensorRTSession:
    """TensorRT 10+ adapter for a serialized engine; device memory via cuda-bindings' runtime."""

    def __init__(self, path: str | Path) -> None:
        try:
            import tensorrt as trt
            from cuda.bindings import runtime as cudart
        except ImportError as exc:
            raise RuntimeError(
                "Install tensorrt-cu12 and cuda-bindings to load a .engine shuttle model"
            ) from exc
        self._cudart = cudart
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(self._logger)
        self._engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self._engine is None or not hasattr(self._engine, "num_io_tensors"):
            raise RuntimeError("TensorRT 10 or newer and a valid serialized engine are required")
        self._context = self._engine.create_execution_context()
        self._stream = _cuda_check(cudart.cudaStreamCreate())
        self._names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        self._inputs = [
            name
            for name in self._names
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self._outputs = [name for name in self._names if name not in self._inputs]
        # name -> (shape, dtype, device pointer)
        self._buffers: dict[str, tuple[tuple[int, ...], np.dtype[Any], int]] = {}

    def get_inputs(self) -> list[TensorInfo]:
        return [
            TensorInfo(name, tuple(self._engine.get_tensor_shape(name))) for name in self._inputs
        ]

    def get_outputs(self) -> list[TensorInfo]:
        return [
            TensorInfo(name, tuple(self._engine.get_tensor_shape(name))) for name in self._outputs
        ]

    def _buffer(self, name: str, shape: tuple[int, ...], dtype: np.dtype[Any]) -> int:
        current = self._buffers.get(name)
        if current is not None and current[0] == shape and current[1] == dtype:
            return current[2]
        if current is not None:
            _cuda_check(self._cudart.cudaFree(current[2]))
        pointer = int(_cuda_check(self._cudart.cudaMalloc(int(np.prod(shape)) * dtype.itemsize)))
        self._buffers[name] = (shape, dtype, pointer)
        return pointer

    def _copy(self, destination: int, source: int, size: int, kind: Any) -> None:
        _cuda_check(self._cudart.cudaMemcpyAsync(destination, source, size, kind, self._stream))

    def run(
        self, output_names: list[str] | None, values: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        kinds = self._cudart.cudaMemcpyKind
        host_inputs: list[np.ndarray] = []  # kept alive until the stream is synchronised
        for name, value in values.items():
            dtype = self._trt.nptype(self._engine.get_tensor_dtype(name))
            host = np.ascontiguousarray(value, dtype=dtype)
            host_inputs.append(host)
            buffer = self._buffer(name, tuple(host.shape), host.dtype)
            self._copy(buffer, host.ctypes.data, host.nbytes, kinds.cudaMemcpyHostToDevice)
            self._context.set_tensor_address(name, buffer)

        outputs: list[tuple[np.ndarray, int]] = []
        for name in output_names or self._outputs:
            shape = tuple(int(dim) for dim in self._context.get_tensor_shape(name))
            dtype = np.dtype(self._trt.nptype(self._engine.get_tensor_dtype(name)))
            host = np.empty(shape, dtype=dtype)
            buffer = self._buffer(name, shape, dtype)
            self._context.set_tensor_address(name, buffer)
            outputs.append((host, buffer))

        if not self._context.execute_async_v3(int(self._stream)):
            raise RuntimeError("TensorRT shuttle inference failed")
        for host, buffer in outputs:
            self._copy(host.ctypes.data, buffer, host.nbytes, kinds.cudaMemcpyDeviceToHost)
        _cuda_check(self._cudart.cudaStreamSynchronize(self._stream))
        return [host for host, _ in outputs]


class TrackNetV3Tracker:
    """Run official 8-frame TrackNetV3 with a background image prepended to RGB frames."""

    def __init__(
        self,
        path: str | Path,
        backend: str,
        *,
        confidence_threshold: float = 0.45,
        morph_kernel_size: int = 3,
        min_contour_area: float = 5.0,
        max_contour_area: float = 120.0,
        session: Session | None = None,
    ) -> None:
        self.path = Path(path)
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.morph_kernel_size = morph_kernel_size
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        # True when a GPU provider was available but the session fell back to CPU.
        self.degraded = False
        self.session = session or self._create_session(backend)

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("TrackNetV3 requires one NCHW input and one NCHW heatmap output")
        input_shape, output_shape = inputs[0].shape, outputs[0].shape
        if len(input_shape) != 4 or len(output_shape) != 4:
            raise ValueError(f"Unsupported TrackNetV3 tensor shapes: {input_shape}, {output_shape}")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.seq_len = self._static_dimension(output_shape[1], "output heatmap count")
        self.height = self._static_dimension(input_shape[2], "input height")
        self.width = self._static_dimension(input_shape[3], "input width")
        input_channels = self._static_dimension(input_shape[1], "input channel count")
        if input_channels != (self.seq_len + 1) * 3:
            raise ValueError(
                "Expected one RGB background plus one RGB tensor per heatmap; "
                f"got {input_channels} input channels for {self.seq_len} heatmaps"
            )
        if tuple(output_shape[-2:]) != tuple(input_shape[-2:]):
            raise ValueError("TrackNetV3 input and output heatmap resolutions must match")
        if morph_kernel_size < 1 or morph_kernel_size % 2 == 0:
            raise ValueError("morph_kernel_size must be a positive odd number")
        if min_contour_area < 0 or max_contour_area < min_contour_area:
            raise ValueError("Contour area limits are invalid")
        # Reused by every inference; the worker runs one inference at a time on one thread.
        self._input = np.empty((1, input_channels, self.height, self.width), dtype=np.float32)
        self._input_background: np.ndarray | None = None

    @staticmethod
    def _static_dimension(value: int | str | None, label: str) -> int:
        if not isinstance(value, (int, np.integer)) or int(value) <= 0:
            raise ValueError(f"TrackNetV3 {label} must be static and positive, got {value!r}")
        return int(value)

    def _create_session(self, backend: str) -> Session:
        if backend == "onnxruntime":
            if self.path.suffix.lower() != ".onnx":
                raise ValueError("ONNX Runtime shuttle backend requires a .onnx model")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("Install ONNX Runtime to load the shuttle model") from exc
            providers = ort.get_available_providers()
            selected = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in providers
                else ["CPUExecutionProvider"]
            )
            session = ort.InferenceSession(str(self.path), providers=selected)
            active = session.get_providers()
            LOGGER.info("Shuttle ONNX Runtime providers: requested=%s active=%s", selected, active)
            # onnxruntime-gpu silently falls back to CPU when CUDA libraries fail to load.
            if selected[0] not in active:
                self.degraded = True
                LOGGER.warning(
                    "Shuttle model requested %s but runs on %s; inference will be slow",
                    selected[0],
                    active,
                )
            return session
        if backend == "tensorrt":
            if self.path.suffix.lower() != ".engine":
                raise ValueError("TensorRT shuttle backend requires a .engine model")
            return TensorRTSession(self.path)
        raise ValueError(f"Unsupported shuttle backend: {backend}")

    def warmup(self) -> None:
        sample = np.zeros((1, (self.seq_len + 1) * 3, self.height, self.width), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: sample})

    def _input_tensor(self, frames: Sequence[ResizedFrame], background: np.ndarray) -> np.ndarray:
        if len(frames) != self.seq_len:
            raise ValueError(f"TrackNetV3 requires {self.seq_len} frames, got {len(frames)}")
        if background.shape != (self.height, self.width, 3) or background.dtype != np.uint8:
            raise ValueError("Background must be resized RGB uint8 at the model input resolution")

        # Official dataset.py prepends the RGB median image, then appends RGB frames in time order.
        # The background only changes every few seconds (a new array each time), so it is
        # rewritten only when a different array arrives; the reference keeps its id unique.
        if background is not self._input_background:
            self._input[0, :3] = _normalised_chw(background)
            self._input_background = background
        for index, frame in enumerate(frames, start=1):
            if frame.chw.shape != (3, self.height, self.width):
                raise ValueError("Cached RGB frame has the wrong model input resolution")
            self._input[0, 3 * index : 3 * index + 3] = frame.chw
        return self._input

    def infer(
        self, frames: Sequence[ResizedFrame], background: np.ndarray
    ) -> tuple[bool, list[float] | None, float | None]:
        values = {self.input_name: self._input_tensor(frames, background)}
        output = self.session.run([self.output_name], values)[0]
        heatmaps = np.asarray(output, dtype=np.float32).squeeze()
        expected = (self.seq_len, self.height, self.width)
        if heatmaps.shape != expected:
            raise ValueError(f"Expected heatmaps shaped {expected}, got {output.shape}")
        # The official TrackNetV3 model includes sigmoid and returns probabilities.
        heatmap = heatmaps[-1]
        finite = np.isfinite(heatmap)
        if not finite.any():
            return False, None, None
        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
        if float(heatmap.min()) < 0.0 or float(heatmap.max()) > 1.0:
            raise ValueError("TrackNetV3 output must contain sigmoid heatmap probabilities")
        peak_confidence = float(heatmap.max())

        mask = (heatmap >= self.confidence_threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, float, float, float]] = []
        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not self.min_contour_area <= area <= self.max_contour_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0.0:
                continue
            contour_mask.fill(0)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            confidence = float(heatmap[contour_mask != 0].max())
            center_x = float(moments["m10"] / moments["m00"])
            center_y = float(moments["m01"] / moments["m00"])
            candidates.append((confidence, area, center_x, center_y))

        if not candidates:
            return False, None, peak_confidence
        confidence, _, x, y = max(candidates, key=lambda candidate: (candidate[0], candidate[1]))
        source = frames[-1]
        px = [
            source.origin_x + (x + 0.5) * source.source_width / self.width,
            source.origin_y + (y + 0.5) * source.source_height / self.height,
        ]
        return True, px, confidence


def create_shuttle_tracker(settings: dict[str, Any]) -> ShuttleTracker:
    return TrackNetV3Tracker(
        path=str(settings["path"]),
        backend=str(settings["backend"]),
        confidence_threshold=float(settings["confidence_threshold"]),
        morph_kernel_size=int(settings["morph_kernel_size"]),
        min_contour_area=float(settings["min_contour_area"]),
        max_contour_area=float(settings["max_contour_area"]),
    )
