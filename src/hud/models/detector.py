"""YOLO person detector interface with ONNX Runtime and TensorRT selections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class Det:
    bbox: list[float]
    conf: float
    track_id: int | None = None
    # COCO-17 [x, y, conf] rows from a pose model; None for a detect-only model.
    keypoints: list[list[float]] | None = None


class Detector(Protocol):
    def infer(self, img: Any) -> list[Det]:
        """Run person detection and ByteTrack, returning image-space boxes."""


class YoloByteTrackDetector:
    """Use Ultralytics' ONNX or TensorRT execution backend and ByteTrack tracker."""

    def __init__(self, path: str, backend: str, input_size: int, confidence: float) -> None:
        model_path = Path(path)
        supported = {"onnxruntime": ".onnx", "tensorrt": ".engine"}
        if backend not in supported:
            raise ValueError(f"Unsupported players model backend: {backend}")
        if model_path.suffix.lower() != supported[backend]:
            raise ValueError(f"Backend {backend} requires a {supported[backend]} model")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install the Ultralytics runtime to load the players model") from exc
        # Task (pose or detect) comes from the exported model's metadata, so the
        # model is chosen by models.players.path alone.
        self._model = YOLO(str(model_path))
        self.input_size = input_size
        self.confidence = confidence

    def infer(self, img: Any) -> list[Det]:
        results = self._model.track(
            source=img,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=self.input_size,
            conf=self.confidence,
            classes=[0],
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None:
            return []
        coords = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(coords)
        keypoints = getattr(results[0], "keypoints", None)
        kpts = keypoints.data.cpu().tolist() if keypoints is not None else [None] * len(coords)
        return [
            Det(
                [float(value) for value in bbox],
                float(conf),
                int(track_id) if track_id is not None else None,
                kpt,
            )
            for bbox, conf, track_id, kpt in zip(coords, confs, ids, kpts, strict=True)
        ]


def create_detector(settings: dict[str, Any]) -> Detector:
    """Build a model using only the selected config backend and model path."""
    return YoloByteTrackDetector(
        path=str(settings["path"]),
        backend=str(settings["backend"]),
        input_size=int(settings["input_size"]),
        confidence=float(settings["confidence"]),
    )
