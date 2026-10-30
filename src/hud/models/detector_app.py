"""Appearance vector (players[].app): chest colour with the court floor removed."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

OPENCV_HUE_MAX = 180


def chest_appearance(
    img_bgr: np.ndarray, bbox: list[float], cfg: dict[str, Any]
) -> list[float] | None:
    """Return a fixed-length vector: hue histogram, then dark and light fractions; sums to 1.

    Pixels inside the configured floor HSV range are dropped first, so floor showing
    through the chest crop does not pull the colour towards the court.
    """
    length = int(cfg["length"])
    height, width = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1
    fx1, fy1, fx2, fy2 = cfg["chest_box"]
    cx1, cx2 = max(0, int(x1 + box_w * fx1)), min(width, int(x1 + box_w * fx2))
    cy1, cy2 = max(0, int(y1 + box_h * fy1)), min(height, int(y1 + box_h * fy2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(img_bgr[cy1:cy2, cx1:cx2]), cv2.COLOR_BGR2HSV)
    floor_low, floor_high = (np.array(bound, dtype=np.uint8) for bound in cfg["floor_hsv"])
    keep = cv2.inRange(hsv, floor_low, floor_high) == 0
    pixels = hsv[keep]
    if len(pixels) == 0:
        return None
    hue, sat, val = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    achromatic = sat <= int(cfg["achromatic_s_max"])
    dark = achromatic & (val <= int(cfg["dark_v_max"]))
    hue_bins = length - 2
    hist = np.bincount(
        hue[~achromatic].astype(np.int32) * hue_bins // OPENCV_HUE_MAX, minlength=hue_bins
    )
    vector = np.concatenate([hist, [dark.sum(), (achromatic & ~dark).sum()]]) / len(pixels)
    return [round(float(value), 4) for value in vector]
