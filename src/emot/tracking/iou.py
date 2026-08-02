"""Vectorized IoU between two sets of `[left, top, right, bottom]` boxes."""

from __future__ import annotations

import numpy as np


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Returns an `(len(boxes_a), len(boxes_b))` IoU matrix. Empty inputs
    return a correctly-shaped empty array rather than raising, since
    tracking loops constantly hit the "no tracks yet" / "no detections
    this frame" edge cases."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    inter_left = np.maximum(a[..., 0], b[..., 0])
    inter_top = np.maximum(a[..., 1], b[..., 1])
    inter_right = np.minimum(a[..., 2], b[..., 2])
    inter_bottom = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.clip(inter_right - inter_left, 0, None)
    inter_h = np.clip(inter_bottom - inter_top, 0, None)
    inter_area = inter_w * inter_h

    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area_a + area_b - inter_area

    return np.where(union > 1e-9, inter_area / union, 0.0)
