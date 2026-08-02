"""DeepSORT: Kalman motion + appearance-embedding association
(Wojke et al., ICIP 2017).

Association combines two signals per (track, detection) pair:
1. A hard Mahalanobis gate on the Kalman filter's predicted position --
   candidates too far from where the motion model expects them are
   rejected outright, regardless of appearance similarity.
2. Among gated candidates, cosine distance between the detection's
   appearance embedding and the track's recent embedding gallery (the
   *minimum* cosine distance across the gallery, not just the last frame,
   following the original paper -- a single bad-angle frame shouldn't
   permanently damage a track's appearance signature).

This implementation uses a single Hungarian assignment stage over the
combined (gated) cost matrix rather than the original paper's cascaded
per-track-age matching -- documented as a simplification in
docs/limitations.md. It is a real difference from the paper (cascade
matching prioritizes recently-seen tracks, which matters most in dense
crowds with many simultaneous occlusions), not a hidden shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from emot.detection.yolo import Detection
from emot.tracking.iou import iou_matrix
from emot.tracking.track import Track, TrackState

MAHALANOBIS_GATE_95PCT_4DOF = 9.4877  # chi-square, 4 DOF, 95th percentile


@dataclass
class DeepSortConfig:
    max_cosine_distance: float = 0.2
    mahalanobis_gate: float = MAHALANOBIS_GATE_95PCT_4DOF
    iou_fallback_threshold: float = 0.3  # for tracks with no appearance history yet
    min_hits: int = 3
    max_age: int = 30


def _cosine_distance_to_gallery(embedding: np.ndarray, gallery: list[np.ndarray]) -> float:
    if not gallery:
        return 1.0
    gallery_arr = np.stack(gallery)
    similarities = gallery_arr @ embedding
    return float(1.0 - similarities.max())


class DeepSortTracker:
    def __init__(self, config: DeepSortConfig | None = None):
        self.config = config or DeepSortConfig()
        self.tracks: list[Track] = []

    def update(
        self, detections: list[Detection], embeddings: np.ndarray
    ) -> list[Track]:
        cfg = self.config
        for track in self.tracks:
            track.predict()

        matches, unmatched_track_idx, unmatched_det_idx = self._match(detections, embeddings)

        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(
                detections[d_idx].xyxy, detections[d_idx].confidence, embeddings[d_idx]
            )
        for i in unmatched_track_idx:
            self.tracks[i].mark_missed()
        for d_idx in unmatched_det_idx:
            self.tracks.append(
                Track(detections[d_idx].xyxy, detections[d_idx].confidence, embeddings[d_idx])
            )

        for track in self.tracks:
            track.confirm_if_ready(cfg.min_hits)

        self.tracks = [t for t in self.tracks if t.time_since_update <= cfg.max_age]

        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]

    def _match(
        self, detections: list[Detection], embeddings: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        cfg = self.config
        n_tracks, n_dets = len(self.tracks), len(detections)
        if n_tracks == 0 or n_dets == 0:
            return [], list(range(n_tracks)), list(range(n_dets))

        det_boxes = np.array([d.xyxy for d in detections])
        cost = np.ones((n_tracks, n_dets))
        gated = np.zeros((n_tracks, n_dets), dtype=bool)

        for i, track in enumerate(self.tracks):
            for j in range(n_dets):
                mdist = track.kalman.mahalanobis_gate(det_boxes[j])
                if mdist > cfg.mahalanobis_gate:
                    continue
                gated[i, j] = True
                cos_dist = _cosine_distance_to_gallery(embeddings[j], track.embeddings)
                cost[i, j] = cos_dist

        row_idx, col_idx = linear_sum_assignment(cost)
        matches, matched_tracks, matched_dets = [], set(), set()
        for r, c in zip(row_idx, col_idx, strict=True):
            if gated[r, c] and cost[r, c] <= cfg.max_cosine_distance:
                matches.append((r, c))
                matched_tracks.add(r)
                matched_dets.add(c)

        unmatched_track_idx = [i for i in range(n_tracks) if i not in matched_tracks]
        unmatched_det_idx = [i for i in range(n_dets) if i not in matched_dets]

        # IoU fallback for the remaining pairs: new tracks (empty gallery)
        # always fail the cosine-distance test above, so without this a
        # track could never be re-confirmed on its very first re-match.
        if unmatched_track_idx and unmatched_det_idx:
            remaining_tracks = [self.tracks[i] for i in unmatched_track_idx]
            remaining_dets = [detections[j] for j in unmatched_det_idx]
            track_boxes = np.array([t.xyxy for t in remaining_tracks])
            rem_det_boxes = np.array([d.xyxy for d in remaining_dets])
            ious = iou_matrix(track_boxes, rem_det_boxes)
            iou_cost = 1.0 - ious
            r2, c2 = linear_sum_assignment(iou_cost)
            still_unmatched_tracks, still_unmatched_dets = set(), set()
            for r, c in zip(r2, c2, strict=True):
                if ious[r, c] >= cfg.iou_fallback_threshold:
                    matches.append((unmatched_track_idx[r], unmatched_det_idx[c]))
                else:
                    still_unmatched_tracks.add(unmatched_track_idx[r])
            unmatched_track_idx = [
                i for i in unmatched_track_idx if i not in {m[0] for m in matches}
            ]
            unmatched_det_idx = [
                j for j in unmatched_det_idx if j not in {m[1] for m in matches}
            ]

        return matches, unmatched_track_idx, unmatched_det_idx
