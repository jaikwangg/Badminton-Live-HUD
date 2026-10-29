"""Four-corner perspective transform from image pixels to court meters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COURT_LENGTH_M = 13.40
COURT_WIDTH_M = 6.10


def compute_homography(corners_px: list[list[float]]) -> list[list[float]]:
    """Compute a 3×3 pixel-to-court transform from clockwise court corners."""
    if len(corners_px) != 4 or any(len(point) != 2 for point in corners_px):
        raise ValueError("Exactly four [x, y] image corners are required")
    source = np.asarray(corners_px, dtype=np.float64)
    if not np.isfinite(source).all() or len(np.unique(source, axis=0)) != 4:
        raise ValueError("Court corners must be finite and distinct")
    edges = np.roll(source, -1, axis=0) - source
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if np.any(np.abs(cross) < 1e-6) or not np.all(cross > 0):
        raise ValueError("Court corners must form a convex quadrilateral in clockwise order")
    target = np.asarray(
        [[0, 0], [COURT_LENGTH_M, 0], [COURT_LENGTH_M, COURT_WIDTH_M], [0, COURT_WIDTH_M]],
        dtype=np.float64,
    )
    rows: list[list[float]] = []
    values: list[float] = []
    for (u, v), (x, y) in zip(source, target, strict=True):
        rows.extend(
            [
                [u, v, 1, 0, 0, 0, -x * u, -x * v],
                [0, 0, 0, u, v, 1, -y * u, -y * v],
            ]
        )
        values.extend([x, y])
    matrix, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)
    transform = np.append(matrix, 1.0).reshape(3, 3)
    if abs(float(np.linalg.det(transform))) < 1e-12:
        raise ValueError("Court corners do not define a valid homography")
    return transform.tolist()


def transform_point(homography: list[list[float]], point_px: list[float]) -> list[float]:
    """Map one image point to court meters."""
    if len(point_px) != 2:
        raise ValueError("Image point must be [x, y]")
    point = np.asarray([point_px[0], point_px[1], 1.0], dtype=np.float64)
    mapped = np.asarray(homography, dtype=np.float64) @ point
    if abs(float(mapped[2])) < 1e-12:
        raise ValueError("Image point maps to infinity")
    return [float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])]


def save_homography(path: str | Path, corners_px: list[list[float]]) -> list[list[float]]:
    """Persist calibration corners and matrix as JSON and return the matrix."""
    homography = compute_homography(corners_px)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"corners_px": corners_px, "homography": homography}, indent=2) + "\n",
        encoding="utf-8",
    )
    return homography


def load_homography(path: str | Path) -> list[list[float]] | None:
    target = Path(path)
    if not target.exists():
        return None
    values = json.loads(target.read_text(encoding="utf-8"))
    return values["homography"]
