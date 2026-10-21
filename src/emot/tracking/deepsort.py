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

Two association modes are implemented so their difference can be measured:

- `cascade=False` (the original baseline of this project): one Hungarian
  assignment over the gated appearance cost for all tracks at once.
- `cascade=True` (the paper's matching cascade): tracks are matched in
  order of how recently they were seen, `age = 1, 2, ..., max_age`, each
  level against only the detections still unclaimed. A track seen one
  frame ago therefore always wins a contested detection over one that
  vanished 20 frames ago -- the prior that matters most in dense crowds
  with many simultaneous occlusions. Leftovers (plus tentative tracks) get
  a final IoU pass, exactly as in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from emot.detection.yolo import Detection
from emot.tracking.iou import iou_matrix
from emot.tracking.kalman import mahalanobis_matrix
from emot.tracking.track import Track, TrackState

MAHALANOBIS_GATE_95PCT_4DOF = 9.4877  # chi-square, 4 DOF, 95th percentile


@dataclass
class DeepSortConfig:
    max_cosine_distance: float = 0.2
    mahalanobis_gate: float = MAHALANOBIS_GATE_95PCT_4DOF
    iou_fallback_threshold: float = 0.3  # for tracks with no appearance history yet
    min_hits: int = 3
    max_age: int = 30
    cascade: bool = False
    min_confidence: float = 0.25  # detections below this are ignored entirely


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
        if len(detections):
            keep = [i for i, d in enumerate(detections) if d.confidence >= cfg.min_confidence]
            if len(keep) != len(detections):
                detections = [detections[i] for i in keep]
                embeddings = embeddings[keep]
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

    def _appearance_cost(
        self, track_idx: list[int], det_idx: list[int], detections: list[Detection],
        embeddings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gated cosine-distance cost for the given track/detection subsets."""
        tracks = [self.tracks[i] for i in track_idx]
        boxes = np.array([detections[j].xyxy for j in det_idx])
        dist = mahalanobis_matrix([t.kalman for t in tracks], boxes)
        gated = dist <= self.config.mahalanobis_gate
        cost = np.ones((len(tracks), len(det_idx)))
        for a, track in enumerate(tracks):
            if not track.embeddings:
                continue
            cols = np.flatnonzero(gated[a])
            if len(cols) == 0:
                continue
            gallery = np.stack(track.embeddings)
            cost[a, cols] = 1.0 - (gallery @ embeddings[[det_idx[c] for c in cols]].T).max(0)
        return cost, gated

    def _hungarian(
        self, cost: np.ndarray, gated: np.ndarray, max_cost: float
    ) -> list[tuple[int, int]]:
        rows, cols = linear_sum_assignment(cost)
        return [
            (r, c)
            for r, c in zip(rows, cols, strict=True)
            if gated[r, c] and cost[r, c] <= max_cost
        ]

    def _match(
        self, detections: list[Detection], embeddings: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        cfg = self.config
        n_tracks, n_dets = len(self.tracks), len(detections)
        if n_tracks == 0 or n_dets == 0:
            return [], list(range(n_tracks)), list(range(n_dets))

        matches: list[tuple[int, int]] = []
        unmatched_dets = list(range(n_dets))

        if cfg.cascade:
            seasoned = [i for i, t in enumerate(self.tracks) if t.state != TrackState.TENTATIVE]
            by_age: dict[int, list[int]] = {}
            for i in seasoned:
                by_age.setdefault(self.tracks[i].time_since_update, []).append(i)
            for age in sorted(by_age):
                if not unmatched_dets:
                    break
                t_idx = by_age[age]
                cost, gated = self._appearance_cost(t_idx, unmatched_dets, detections, embeddings)
                found = self._hungarian(cost, gated, cfg.max_cosine_distance)
                matches += [(t_idx[r], unmatched_dets[c]) for r, c in found]
                taken = {c for _, c in found}
                unmatched_dets = [d for k, d in enumerate(unmatched_dets) if k not in taken]
            matched_tracks = {t for t, _ in matches}
            # IoU stage: tentative tracks and confirmed tracks seen just one frame ago
            iou_pool = [
                i for i, t in enumerate(self.tracks)
                if i not in matched_tracks
                and (t.state == TrackState.TENTATIVE or t.time_since_update == 1)
            ]
        else:
            all_tracks = list(range(n_tracks))
            cost, gated = self._appearance_cost(all_tracks, unmatched_dets, detections, embeddings)
            matches = self._hungarian(cost, gated, cfg.max_cosine_distance)
            matched_tracks = {t for t, _ in matches}
            matched_dets = {d for _, d in matches}
            unmatched_dets = [d for d in range(n_dets) if d not in matched_dets]
            # IoU fallback: new tracks (empty gallery) always fail the cosine test above,
            # so without this a track could never be re-confirmed on its first re-match.
            iou_pool = [i for i in range(n_tracks) if i not in matched_tracks]

        if iou_pool and unmatched_dets:
            track_boxes = np.array([self.tracks[i].xyxy for i in iou_pool])
            det_boxes = np.array([detections[j].xyxy for j in unmatched_dets])
            ious = iou_matrix(track_boxes, det_boxes)
            rows, cols = linear_sum_assignment(1.0 - ious)
            used_d = set()
            for r, c in zip(rows, cols, strict=True):
                if ious[r, c] >= cfg.iou_fallback_threshold:
                    matches.append((iou_pool[r], unmatched_dets[c]))
                    used_d.add(unmatched_dets[c])
            unmatched_dets = [d for d in unmatched_dets if d not in used_d]

        matched_t = {t for t, _ in matches}
        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_t]
        return matches, unmatched_tracks, unmatched_dets
