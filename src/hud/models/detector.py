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


# ponytail: fixed constant, not config; move to models.players (T01 owns config) to tune.
TILE_EDGE_PX = 2.0  # a tile box this close to an edge inside the frame is a cut-off person


def _iou(a: list[float], b: list[float]) -> float:
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def merge_duplicates(detections: list[Det], iou: float) -> list[Det]:
    """Greedy NMS across tiles: keep the most confident box of each overlapping group."""
    kept: list[Det] = []
    for det in sorted(detections, key=lambda d: d.conf, reverse=True):
        if all(_iou(det.bbox, other.bbox) < iou for other in kept):
            kept.append(det)
    return kept


class YoloByteTrackDetector:
    """Use Ultralytics' ONNX or TensorRT execution backend and ByteTrack tracker."""

    def __init__(
        self,
        path: str,
        backend: str,
        input_size: int,
        confidence: float,
        tiles: list[list[float]] | None = None,
        tile_merge_iou: float = 0.5,
    ) -> None:
        model_path = Path(path)
        supported = {"onnxruntime": ".onnx", "tensorrt": ".engine"}
        if backend not in supported:
            raise ValueError(f"Unsupported players model backend: {backend}")
        if model_path.suffix.lower() != supported[backend]:
            raise ValueError(f"Backend {backend} requires a {supported[backend]} model")
        for tile in tiles or []:
            if len(tile) not in (4, 5) or not (
                0 <= tile[0] < tile[2] <= 1 and 0 <= tile[1] < tile[3] <= 1
            ):
                raise ValueError(f"players tile must be [x1, y1, x2, y2(, conf)] fractions: {tile}")
        # None or [] keeps the single full-frame pass with Ultralytics' own tracking.
        self.tiles = tiles or None
        self.tile_merge_iou = tile_merge_iou
        self._tracker: Any = None
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
        if self.tiles is None:
            return self._track_full_frame(img)
        return self._track(img, merge_duplicates(self._detect_tiles(img), self.tile_merge_iou))

    def _track_full_frame(self, img: Any) -> list[Det]:
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

    def _detect_tiles(self, img: Any) -> list[Det]:
        """Detect on every tile crop in one batched call; return frame-space detections."""
        height, width = img.shape[:2]
        crops = []
        for tile in self.tiles or []:
            x0, y0 = int(tile[0] * width), int(tile[1] * height)
            x1, y1 = int(tile[2] * width), int(tile[3] * height)
            conf = float(tile[4]) if len(tile) > 4 else self.confidence
            # Edges inside the frame: a box touching one is a person cut by the tile.
            inner = (x0 > 0, y0 > 0, x1 < width, y1 < height)
            crops.append((img[y0:y1, x0:x1], x0, y0, conf, inner))
        # Ultralytics runs the list in chunks of a static engine's batch size.
        results = self._model.predict(
            [crop[0] for crop in crops],
            imgsz=self.input_size,
            conf=min(crop[3] for crop in crops),
            classes=[0],
            verbose=False,
        )
        detections: list[Det] = []
        for (crop, x0, y0, conf, inner), result in zip(crops, results, strict=True):
            boxes = result.boxes
            if boxes is None:
                continue
            crop_h, crop_w = crop.shape[:2]
            keypoints = getattr(result, "keypoints", None)
            kpts = keypoints.data.cpu().tolist() if keypoints is not None else None
            coords = boxes.xyxy.cpu().tolist()
            for i, (bbox, score) in enumerate(zip(coords, boxes.conf.cpu().tolist(), strict=True)):
                cut = (
                    bbox[0] <= TILE_EDGE_PX,
                    bbox[1] <= TILE_EDGE_PX,
                    bbox[2] >= crop_w - TILE_EDGE_PX,
                    bbox[3] >= crop_h - TILE_EDGE_PX,
                )
                # Overlapping tiles (or a full-frame tile) see a cut person whole.
                if score < conf or any(c and e for c, e in zip(cut, inner, strict=True)):
                    continue
                detections.append(
                    Det(
                        [bbox[0] + x0, bbox[1] + y0, bbox[2] + x0, bbox[3] + y0],
                        float(score),
                        None,
                        [[x + x0, y + y0, c] for x, y, c in kpts[i]] if kpts is not None else None,
                    )
                )
        return detections

    def _track(self, img: Any, detections: list[Det]) -> list[Det]:
        """Give merged tile detections ByteTrack IDs with one tracker for the whole frame."""
        import numpy as np
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        if self._tracker is None:
            settings = YAML.load(check_yaml("bytetrack.yaml"))
            self._tracker = BYTETracker(IterableSimpleNamespace(**settings))
        rows = np.array([[*d.bbox, d.conf, 0.0] for d in detections], dtype=np.float32)
        tracks = self._tracker.update(Boxes(rows.reshape(-1, 6), img.shape[:2]), img)
        # As in the full-frame path: tracker box, keypoints of the matched detection (row[7]).
        return [
            Det(
                [float(value) for value in row[:4]],
                float(row[5]),
                int(row[4]),
                detections[int(row[7])].keypoints,
            )
            for row in tracks
        ]


def create_detector(settings: dict[str, Any]) -> Detector:
    """Build a model using only the selected config backend and model path."""
    return YoloByteTrackDetector(
        path=str(settings["path"]),
        backend=str(settings["backend"]),
        input_size=int(settings["input_size"]),
        confidence=float(settings["confidence"]),
        tiles=settings.get("tiles"),
        tile_merge_iou=float(settings.get("tile_merge_iou", 0.5)),
    )
