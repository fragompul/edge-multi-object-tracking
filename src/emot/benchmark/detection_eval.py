"""Scores a cached detection run against MOT17 ground truth (COCO-style AP)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from emot.benchmark.caches import CachedDetections
from emot.evaluation.detection_ap import DetectionAP, evaluate_detections

# Persons visible in less than a quarter of their box are scored as "ignore regions".
MIN_EVAL_VISIBILITY = 0.25


def evaluate_cached(cached: CachedDetections, gt_df: pd.DataFrame) -> DetectionAP:
    gt_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame, rows in gt_df.groupby("frame"):
        boxes = rows[["bb_left", "bb_top", "bb_right", "bb_bottom"]].to_numpy(float)
        ignore = (rows["visibility"] < MIN_EVAL_VISIBILITY).to_numpy()
        gt_by_frame[int(frame)] = (boxes, ignore)  # type: ignore[call-overload]

    det_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame in np.unique(cached.frame_id):
        rows = cached.rows(int(frame))
        det_by_frame[int(frame)] = (cached.boxes[rows], cached.conf[rows])
    return evaluate_detections(det_by_frame, gt_by_frame)
