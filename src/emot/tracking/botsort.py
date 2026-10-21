"""BoT-SORT-style tracker (Aharon et al., 2022): ByteTrack's two-stage
association, plus the two upgrades that matter on real footage --

1. **Camera-motion compensation (CMC):** every track is warped by the
   estimated global image motion before matching, so a panning camera no
   longer reads as every pedestrian accelerating at once.
2. **Optional appearance fusion (ReID):** the stage-1 cost is
   `min(IoU distance, cosine distance / 2)`, gated so appearance can only
   rescue pairs that are already spatially plausible and only when it is
   confident. Track appearance is an exponential moving average, which is
   far cheaper and more stable than keeping a gallery.

Because the geometry, thresholds and track lifecycle are *shared* with
`ByteTracker`, switching `use_cmc` / `use_reid` on and off is a clean
ablation: `use_cmc=False, use_reid=False` is exactly ByteTrack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from emot.detection.yolo import Detection
from emot.tracking.iou import iou_matrix
from emot.tracking.track import Track, TrackState


@dataclass
class BotSortConfig:
    high_conf_threshold: float = 0.6
    low_conf_threshold: float = 0.1
    match_iou_threshold: float = 0.8  # max association distance in stage 1 (see ByteTrackConfig)
    low_match_iou_threshold: float = 0.5
    min_hits: int = 3
    max_age: int = 30
    use_cmc: bool = True
    use_reid: bool = False
    proximity_threshold: float = 0.5  # appearance is ignored beyond this IoU distance
    appearance_threshold: float = 0.25  # ... and beyond this (halved) cosine distance
    ema_alpha: float = 0.9


class _BotTrack(Track):
    """Track with an EMA appearance descriptor instead of a gallery."""

    def __init__(self, xyxy: np.ndarray, confidence: float, embedding: np.ndarray | None):
        super().__init__(xyxy, confidence)
        self.smooth_embedding = embedding

    def update_embedding(self, embedding: np.ndarray | None, alpha: float) -> None:
        if embedding is None:
            return
        if self.smooth_embedding is None:
            self.smooth_embedding = embedding
            return
        mixed = alpha * self.smooth_embedding + (1.0 - alpha) * embedding
        self.smooth_embedding = mixed / max(float(np.linalg.norm(mixed)), 1e-8)


def _assign(
    cost: np.ndarray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches = [(r, c) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= max_cost]
    used_r, used_c = {r for r, _ in matches}, {c for _, c in matches}
    return (
        matches,
        [i for i in range(cost.shape[0]) if i not in used_r],
        [j for j in range(cost.shape[1]) if j not in used_c],
    )


class BotSortTracker:
    def __init__(self, config: BotSortConfig | None = None):
        self.config = config or BotSortConfig()
        self.tracks: list[_BotTrack] = []

    def _stage1_cost(
        self, tracks: list[_BotTrack], dets: list[Detection], det_embeddings: np.ndarray | None
    ) -> np.ndarray:
        cfg = self.config
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        iou_dist = 1.0 - iou_matrix(
            np.array([t.xyxy for t in tracks]), np.array([d.xyxy for d in dets])
        )
        if not cfg.use_reid or det_embeddings is None:
            return iou_dist
        has_emb = np.array([t.smooth_embedding is not None for t in tracks])
        zero = np.zeros_like(det_embeddings[0])
        track_emb = np.stack(
            [zero if t.smooth_embedding is None else t.smooth_embedding for t in tracks]
        )
        emb_dist = np.clip(1.0 - track_emb @ det_embeddings.T, 0.0, 2.0) / 2.0
        emb_dist[emb_dist > cfg.appearance_threshold] = 1.0
        emb_dist[iou_dist > cfg.proximity_threshold] = 1.0
        emb_dist[~has_emb] = 1.0
        return np.minimum(iou_dist, emb_dist)

    def update(
        self,
        detections: list[Detection],
        embeddings: np.ndarray | None = None,
        warp: np.ndarray | None = None,
    ) -> list[Track]:
        cfg = self.config
        for track in self.tracks:
            track.predict()
            if cfg.use_cmc and warp is not None:
                track.kalman.apply_affine(warp)

        high_idx = [i for i, d in enumerate(detections) if d.confidence >= cfg.high_conf_threshold]
        low_idx = [
            i
            for i, d in enumerate(detections)
            if cfg.low_conf_threshold <= d.confidence < cfg.high_conf_threshold
        ]
        high = [detections[i] for i in high_idx]
        low = [detections[i] for i in low_idx]
        high_emb = embeddings[high_idx] if embeddings is not None and len(high_idx) else None

        # Stage 1: every track vs high-confidence detections (IoU, optionally + appearance).
        cost1 = self._stage1_cost(self.tracks, high, high_emb)
        matches1, unmatched_t, unmatched_high = _assign(cost1, cfg.match_iou_threshold)
        for t_idx, d_idx in matches1:
            track = self.tracks[t_idx]
            track.update(high[d_idx].xyxy, high[d_idx].confidence)
            if high_emb is not None:
                track.update_embedding(high_emb[d_idx], cfg.ema_alpha)

        # Stage 2: leftover tracks vs low-confidence detections (IoU only, never starts tracks).
        remaining = [self.tracks[i] for i in unmatched_t]
        if remaining and low:
            iou_dist = 1.0 - iou_matrix(
                np.array([t.xyxy for t in remaining]), np.array([d.xyxy for d in low])
            )
            matches2, still_unmatched, _ = _assign(iou_dist, cfg.low_match_iou_threshold)
            for t_idx, d_idx in matches2:
                remaining[t_idx].update(low[d_idx].xyxy, low[d_idx].confidence)
        else:
            still_unmatched = list(range(len(remaining)))
        for i in still_unmatched:
            remaining[i].mark_missed()

        for d_idx in unmatched_high:
            emb = high_emb[d_idx] if high_emb is not None else None
            self.tracks.append(_BotTrack(high[d_idx].xyxy, high[d_idx].confidence, emb))

        for track in self.tracks:
            track.confirm_if_ready(cfg.min_hits)
        self.tracks = [t for t in self.tracks if t.time_since_update <= cfg.max_age]
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]
