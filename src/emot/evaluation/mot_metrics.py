"""Wraps `py-motmetrics` (the standard CLEAR-MOT + IDF1 implementation)
instead of reimplementing these metrics from scratch.

Unlike this portfolio's MVTec AD project (where writing PRO/pixel-AUROC
by hand was reasonable and in fact caught a real bug), CLEAR-MOT metrics
have many easy-to-get-subtly-wrong edge cases (frame-to-frame ID-switch
bookkeeping, mostly-tracked/mostly-lost thresholds, distance-normalized
matching) that the field has already standardized around one reference
implementation -- reimplementing them here would only add a second place
for those edge cases to silently diverge from the standard, with no
compensating engineering value.
"""

from __future__ import annotations

from dataclasses import dataclass

import motmetrics as mm
import numpy as np
import pandas as pd

from emot.evaluation.hota import HOTAAccumulator
from emot.tracking.iou import iou_matrix
from emot.tracking.track import Track


@dataclass(frozen=True)
class SequenceMetrics:
    mota: float
    motp: float
    idf1: float
    num_switches: int
    num_false_positives: int
    num_misses: int
    mostly_tracked: int
    mostly_lost: int
    num_frames: int
    # HOTA family (Luiten et al., 2021): detection / association / localisation split
    hota: float = 0.0
    deta: float = 0.0
    assa: float = 0.0
    loca: float = 0.0


def _xyxy_to_ltwh(xyxy: np.ndarray) -> np.ndarray:
    left, top, right, bottom = xyxy
    return np.array([left, top, right - left, bottom - top])


def _iou_distance_matrix(
    gt_boxes_ltwh: np.ndarray, pred_boxes_ltwh: np.ndarray, max_iou_distance: float
) -> np.ndarray:
    """Reimplements `motmetrics.distances.iou_matrix` in terms of this
    project's own IoU utility, instead of calling the library function
    directly.

    Found while wiring this up: `py-motmetrics` 1.4.0's `iou_matrix` calls
    `np.asfarray`, which was removed in NumPy 2.0 (this project pins
    numpy>=1.26 with no upper bound, and the installed resolved version is
    2.x) -- every call raises `AttributeError` immediately, so this isn't
    a subtle numerical issue, it's a hard crash on the very first
    evaluation. Reimplementing the ~10 lines of IoU-to-distance logic
    locally (reusing the already-tested `emot.tracking.iou.iou_matrix`)
    avoids depending on an unmaintained upstream fix landing, and reuses
    code this project already needs for tracking anyway.
    """
    if len(gt_boxes_ltwh) == 0 or len(pred_boxes_ltwh) == 0:
        return np.empty((len(gt_boxes_ltwh), len(pred_boxes_ltwh)))

    def ltwh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        left, top, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return np.stack([left, top, left + width, top + height], axis=1)

    ious = iou_matrix(ltwh_to_xyxy(gt_boxes_ltwh), ltwh_to_xyxy(pred_boxes_ltwh))
    dist = 1.0 - ious
    return np.where(dist > max_iou_distance, np.nan, dist)


class MOTAccumulator:
    """Thin per-sequence wrapper: call `update` once per frame with
    ground-truth boxes and predicted tracks, then `compute` at the end."""

    def __init__(self, max_iou_distance: float = 0.5):
        self.acc = mm.MOTAccumulator(auto_id=True)
        self.hota = HOTAAccumulator()
        self.max_iou_distance = max_iou_distance

    def update(self, gt_array: np.ndarray, predicted_tracks: list[Track]) -> None:
        self.update_arrays(
            gt_array,
            [t.track_id for t in predicted_tracks],
            np.array([t.xyxy for t in predicted_tracks]) if predicted_tracks else np.zeros((0, 4)),
        )

    def update_arrays(
        self, gt_array: np.ndarray, pred_ids: list[int], pred_xyxy: np.ndarray
    ) -> None:
        """Same as `update`, for callers holding plain ids + `[l, t, r, b]` boxes."""
        gt_ids = gt_array[:, 0].astype(int).tolist() if len(gt_array) else []
        gt_boxes = gt_array[:, 1:5] if len(gt_array) else np.zeros((0, 4))
        pred_boxes = (
            np.array([_xyxy_to_ltwh(b) for b in pred_xyxy]) if len(pred_xyxy) else np.zeros((0, 4))
        )
        distances = _iou_distance_matrix(gt_boxes, pred_boxes, self.max_iou_distance)
        self.acc.update(gt_ids, pred_ids, distances)
        self.hota.update(np.array(gt_ids), gt_boxes, np.array(pred_ids), pred_boxes)

    def switch_events(self) -> list[dict[str, int]]:
        """Every identity switch as `{frame, gt_id, from_track, to_track}` (1-based frames):
        ground-truth person `gt_id` was matched to `from_track` and is now matched to
        `to_track`. Used to place switch markers on the demo's timelines."""
        events = self.acc.mot_events
        out = []
        last_match: dict[int, int] = {}
        for (frame_id, _), row in events.iterrows():
            if row["Type"] in ("MATCH", "SWITCH"):
                gt_id, track = int(row["OId"]), int(row["HId"])
                if row["Type"] == "SWITCH":
                    out.append({
                        "frame": int(frame_id) + 1,
                        "gt_id": gt_id,
                        "from_track": last_match.get(gt_id, -1),
                        "to_track": track,
                    })
                last_match[gt_id] = track
        return out

    def compute(self, sequence_name: str) -> SequenceMetrics:
        metrics_host = mm.metrics.create()
        summary = metrics_host.compute(
            self.acc,
            metrics=[
                "mota",
                "motp",
                "idf1",
                "num_switches",
                "num_false_positives",
                "num_misses",
                "mostly_tracked",
                "mostly_lost",
                "num_frames",
            ],
            name=sequence_name,
        )
        row = summary.iloc[0]
        hota = self.hota.compute()
        return SequenceMetrics(
            mota=float(row["mota"]),
            motp=float(row["motp"]) if not pd.isna(row["motp"]) else 0.0,
            idf1=float(row["idf1"]),
            num_switches=int(row["num_switches"]),
            num_false_positives=int(row["num_false_positives"]),
            num_misses=int(row["num_misses"]),
            mostly_tracked=int(row["mostly_tracked"]),
            mostly_lost=int(row["mostly_lost"]),
            num_frames=int(row["num_frames"]),
            hota=hota.hota,
            deta=hota.deta,
            assa=hota.assa,
            loca=hota.loca,
        )
