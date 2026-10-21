"""HOTA (Luiten et al., "HOTA: A Higher Order Metric for Evaluating
Multi-Object Tracking", IJCV 2021), implemented from the paper and
cross-checked numerically against the reference `TrackEval` code (see
`docs/experiments.md`).

Why HOTA on top of MOTA/IDF1: MOTA is dominated by detection errors and
barely sees identity errors; IDF1 is dominated by association and barely
sees localisation. HOTA decomposes the score into the two halves the user
actually cares about and reports their geometric mean:

    HOTA_a = sqrt(DetA_a * AssA_a),   HOTA = mean over a in {0.05, ..., 0.95}

- DetA: "did you find the people?" -- Jaccard of detections, TP/(TP+FN+FP)
- AssA: "did you keep their identities straight?" -- averaged over TPs,
  each scored by the Jaccard of its (gt-id, pred-id) trajectory pair
- LocA: mean IoU of matched pairs (localisation quality)

Averaging over 19 localisation thresholds rewards precise boxes without
hard-coding one IoU cut-off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from emot.tracking.iou import iou_matrix

ALPHAS = np.arange(0.05, 0.99, 0.05)  # 19 thresholds, as in TrackEval
_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class HOTAScores:
    hota: float
    deta: float
    assa: float
    loca: float
    det_re: float
    det_pr: float
    ass_re: float
    ass_pr: float


class HOTAAccumulator:
    """Per-frame accumulation, then a single `compute()` over the sequence.

    `update` takes boxes as `[left, top, width, height]` (MOTChallenge layout)
    plus integer identities for both ground truth and predictions.
    """

    def __init__(self) -> None:
        self._gt_ids: list[np.ndarray] = []
        self._pred_ids: list[np.ndarray] = []
        self._sims: list[np.ndarray] = []

    @staticmethod
    def _to_xyxy(boxes: np.ndarray) -> np.ndarray:
        return np.stack(
            [boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]],
            axis=1,
        )

    def update(
        self,
        gt_ids: np.ndarray,
        gt_boxes_ltwh: np.ndarray,
        pred_ids: np.ndarray,
        pred_boxes_ltwh: np.ndarray,
    ) -> None:
        gt_ids = np.asarray(gt_ids, dtype=np.int64)
        pred_ids = np.asarray(pred_ids, dtype=np.int64)
        if len(gt_ids) and len(pred_ids):
            sim = iou_matrix(self._to_xyxy(gt_boxes_ltwh), self._to_xyxy(pred_boxes_ltwh))
        else:
            sim = np.zeros((len(gt_ids), len(pred_ids)))
        self._gt_ids.append(gt_ids)
        self._pred_ids.append(pred_ids)
        self._sims.append(sim)

    def compute(self) -> HOTAScores:
        n_alpha = len(ALPHAS)
        no_ids = np.array([], dtype=np.int64)
        all_gt = np.unique(np.concatenate(self._gt_ids)) if self._gt_ids else no_ids
        all_pred = np.unique(np.concatenate(self._pred_ids)) if self._pred_ids else no_ids
        n_gt, n_pred = len(all_gt), len(all_pred)
        if n_gt == 0:
            return HOTAScores(*(0.0,) * 8)
        if n_pred == 0:
            return HOTAScores(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

        gt_frames = [np.searchsorted(all_gt, ids) for ids in self._gt_ids]
        pred_frames = [np.searchsorted(all_pred, ids) for ids in self._pred_ids]

        # Pass 1: global alignment score between every (gt-id, pred-id) trajectory pair.
        potential = np.zeros((n_gt, n_pred))
        gt_count = np.zeros(n_gt)
        pred_count = np.zeros(n_pred)
        for g, p, sim in zip(gt_frames, pred_frames, self._sims, strict=True):
            gt_count[g] += 1
            pred_count[p] += 1
            if len(g) and len(p):
                denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
                sim_iou = np.divide(sim, denom, out=np.zeros_like(sim), where=denom > _EPS)
                potential[np.ix_(g, p)] += sim_iou
        global_align = potential / (gt_count[:, None] + pred_count[None, :] - potential)

        # Pass 2: per-frame Hungarian matching on alignment-weighted similarity.
        tp = np.zeros(n_alpha)
        fn = np.zeros(n_alpha)
        fp = np.zeros(n_alpha)
        loc_sum = np.zeros(n_alpha)
        matches = np.zeros((n_alpha, n_gt, n_pred))
        for g, p, sim in zip(gt_frames, pred_frames, self._sims, strict=True):
            if len(g) == 0:
                fp += len(p)
                continue
            if len(p) == 0:
                fn += len(g)
                continue
            score = global_align[np.ix_(g, p)] * sim
            rows, cols = linear_sum_assignment(-score)
            matched_sim = sim[rows, cols]
            for a, alpha in enumerate(ALPHAS):
                ok = matched_sim >= alpha - _EPS
                n_ok = int(ok.sum())
                tp[a] += n_ok
                fn[a] += len(g) - n_ok
                fp[a] += len(p) - n_ok
                loc_sum[a] += matched_sim[ok].sum()
                matches[a][g[rows[ok]], p[cols[ok]]] += 1

        gt_c = gt_count[None, :, None]
        pred_c = pred_count[None, None, :]
        ass_jac = matches / np.maximum(1.0, gt_c + pred_c - matches)
        safe_tp = np.maximum(1.0, tp)
        assa = (matches * ass_jac).sum((1, 2)) / safe_tp
        ass_re = (matches * (matches / np.maximum(1.0, gt_c))).sum((1, 2)) / safe_tp
        ass_pr = (matches * (matches / np.maximum(1.0, pred_c))).sum((1, 2)) / safe_tp
        deta = tp / np.maximum(1.0, tp + fn + fp)
        det_re = tp / np.maximum(1.0, tp + fn)
        det_pr = tp / np.maximum(1.0, tp + fp)
        loca = np.where(tp > 0, loc_sum / safe_tp, 1.0)
        hota = np.sqrt(deta * assa)
        return HOTAScores(
            hota=float(hota.mean()),
            deta=float(deta.mean()),
            assa=float(assa.mean()),
            loca=float(loca.mean()),
            det_re=float(det_re.mean()),
            det_pr=float(det_pr.mean()),
            ass_re=float(ass_re.mean()),
            ass_pr=float(ass_pr.mean()),
        )
