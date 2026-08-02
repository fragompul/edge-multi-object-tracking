"""Shared logic for replaying cached per-frame detections through a
tracker and scoring the result against ground truth -- used by both the
hyperparameter tuning script (many short runs on a val split) and the
full benchmark script (fewer, full-sequence runs), so the two can never
silently diverge in how a track is scored.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from emot.detection.yolo import Detection
from emot.evaluation.mot_metrics import MOTAccumulator, SequenceMetrics
from emot.tracking.bytetrack import ByteTrackConfig, ByteTracker
from emot.tracking.deepsort import DeepSortConfig, DeepSortTracker
from emot.tracking.reid import AppearanceEmbedder


def load_cached_detections(cache_path: Path) -> dict[int, list[Detection]]:
    data = np.load(cache_path)
    by_frame: dict[int, list[Detection]] = {}
    for frame_id, x1, y1, x2, y2, conf in zip(
        data["frame_id"], data["x1"], data["y1"], data["x2"], data["y2"], data["conf"], strict=True
    ):
        by_frame.setdefault(int(frame_id), []).append(
            Detection(xyxy=np.array([x1, y1, x2, y2], dtype=float), confidence=float(conf))
        )
    return by_frame


def _crop(image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (
        max(0, int(xyxy[0])),
        max(0, int(xyxy[1])),
        min(w, int(xyxy[2])),
        min(h, int(xyxy[3])),
    )
    if x2 <= x1 or y2 <= y1:
        return np.zeros((4, 4, 3), dtype=np.uint8)
    return cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def run_bytetrack(
    detections_by_frame: dict[int, list[Detection]],
    gt_df,
    frame_indices: range,
    config: ByteTrackConfig,
) -> SequenceMetrics:
    from emot.data.mot17 import gt_to_mot_array

    tracker = ByteTracker(config)
    accumulator = MOTAccumulator()
    for frame_idx in frame_indices:
        dets = detections_by_frame.get(frame_idx, [])
        tracks = tracker.update(dets)
        gt_array = gt_to_mot_array(gt_df, frame_idx)
        accumulator.update(gt_array, tracks)
    return accumulator.compute("bytetrack")


def run_deepsort(
    detections_by_frame: dict[int, list[Detection]],
    gt_df,
    frame_indices: range,
    config: DeepSortConfig,
    seq_dir: Path,
    embedder: AppearanceEmbedder,
) -> SequenceMetrics:
    from emot.data.mot17 import frame_path, gt_to_mot_array

    tracker = DeepSortTracker(config)
    accumulator = MOTAccumulator()
    for frame_idx in frame_indices:
        dets = detections_by_frame.get(frame_idx, [])
        image = cv2.imread(str(frame_path(seq_dir, frame_idx)))
        crops = [_crop(image, d.xyxy) for d in dets] if dets else []
        embeddings = embedder.embed_crops(crops)
        tracks = tracker.update(dets, embeddings)
        gt_array = gt_to_mot_array(gt_df, frame_idx)
        accumulator.update(gt_array, tracks)
    return accumulator.compute("deepsort")
