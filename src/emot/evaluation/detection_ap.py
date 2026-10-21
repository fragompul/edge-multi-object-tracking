"""Box-level detection quality (COCO-style average precision) for the
person class, scored directly against MOT17 ground truth.

The first version of this benchmark used downstream tracking MOTA as its
only "accuracy" axis, which conflates detector quality with tracker
behaviour. AP isolates the detector: it asks only "are the boxes right,
ranked by confidence?", independent of any association logic.

Conventions (COCO / PASCAL-101-point hybrid):
- AP is the area under the monotone precision envelope sampled at 101
  recall points, averaged over IoU thresholds 0.50:0.05:0.95 (`ap`) or taken
  at 0.50 (`ap50`).
- Ground-truth persons below `min_visibility` are *ignore regions*: a
  detection that best-matches one is neither a true nor a false positive,
  so a detector is not punished for (not) finding a person who is 90%
  hidden behind a pillar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emot.tracking.iou import iou_matrix

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True)
class DetectionAP:
    ap50: float
    ap: float
    recall50: float  # max recall reached at IoU 0.5 (limited by the cache's confidence floor)
    num_gt: int
    num_detections: int


def _match_frame(
    det_boxes: np.ndarray,
    det_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_ignore: np.ndarray,
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy COCO matching for one frame. Returns per-detection (is_tp, is_ignored),
    both aligned with `det_boxes` *in descending-score order*."""
    order = np.argsort(-det_scores, kind="stable")
    n_det = len(order)
    is_tp = np.zeros(n_det, dtype=bool)
    ignored = np.zeros(n_det, dtype=bool)
    if len(gt_boxes) == 0 or n_det == 0:
        return is_tp, ignored
    # regular ground truth first, ignore regions last (COCO's matching order)
    gt_order = np.argsort(gt_ignore, kind="stable")
    gt_boxes, gt_ignore = gt_boxes[gt_order], gt_ignore[gt_order]
    ious = iou_matrix(det_boxes[order], gt_boxes)
    gt_taken = np.zeros(len(gt_boxes), dtype=bool)
    for d in range(n_det):
        best, best_iou = -1, iou_thr - 1e-10
        for g in range(len(gt_boxes)):
            if gt_taken[g]:
                continue
            # once a regular GT is matched, never downgrade to an ignore region
            if best > -1 and not gt_ignore[best] and gt_ignore[g]:
                break
            if ious[d, g] < best_iou:
                continue
            best_iou, best = ious[d, g], g
        if best == -1:
            continue
        gt_taken[best] = True
        if gt_ignore[best]:
            ignored[d] = True
        else:
            is_tp[d] = True
    return is_tp, ignored


def _ap_from_flags(
    scores: np.ndarray, is_tp: np.ndarray, ignored: np.ndarray, num_gt: int
) -> tuple[float, float]:
    keep = ~ignored
    scores, is_tp = scores[keep], is_tp[keep]
    if num_gt == 0:
        return 0.0, 0.0
    if len(scores) == 0:
        return 0.0, 0.0
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(~is_tp[order])
    recall = tp / num_gt
    precision = tp / np.maximum(tp + fp, 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # monotone envelope
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    sampled = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(sampled.mean()), float(recall[-1])


def evaluate_detections(
    detections_by_frame: dict[int, tuple[np.ndarray, np.ndarray]],
    gt_by_frame: dict[int, tuple[np.ndarray, np.ndarray]],
) -> DetectionAP:
    """`detections_by_frame[f] = (xyxy (N,4), scores (N,))`;
    `gt_by_frame[f] = (xyxy (M,4), ignore_flags (M,) bool)`."""
    empty = (np.zeros((0, 4)), np.zeros(0))
    frames = sorted(set(gt_by_frame) | set(detections_by_frame))
    num_gt = int(sum((~gt_by_frame[f][1]).sum() for f in gt_by_frame))
    num_dets = int(sum(len(detections_by_frame[f][1]) for f in detections_by_frame))

    aps = []
    recall50 = 0.0
    for t_idx, thr in enumerate(IOU_THRESHOLDS):
        all_scores, all_tp, all_ign = [], [], []
        for f in frames:
            boxes, scores = detections_by_frame.get(f, empty)
            gt_boxes, gt_ign = gt_by_frame.get(f, (np.zeros((0, 4)), np.zeros(0, bool)))
            order = np.argsort(-scores, kind="stable")
            tp, ign = _match_frame(boxes, scores, gt_boxes, gt_ign, float(thr))
            all_scores.append(scores[order])
            all_tp.append(tp)
            all_ign.append(ign)
        ap, rec = _ap_from_flags(
            np.concatenate(all_scores) if all_scores else np.zeros(0),
            np.concatenate(all_tp) if all_tp else np.zeros(0, bool),
            np.concatenate(all_ign) if all_ign else np.zeros(0, bool),
            num_gt,
        )
        aps.append(ap)
        if t_idx == 0:
            recall50 = rec
    return DetectionAP(
        ap50=aps[0],
        ap=float(np.mean(aps)),
        recall50=recall50,
        num_gt=num_gt,
        num_detections=num_dets,
    )
