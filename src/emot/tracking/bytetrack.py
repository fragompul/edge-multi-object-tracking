"""ByteTrack: motion-only multi-object tracking (Zhang et al., ECCV 2022).

The key idea BYTE contributes over plain SORT is using *low-confidence*
detections too, instead of discarding them: a box the detector is unsure
about is often a real object under partial occlusion or motion blur, not
noise -- discarding it outright (as SORT does) causes track fragmentation
exactly when a person is briefly occluded, which is the single most common
real-world failure mode in pedestrian tracking. BYTE recovers these by
running a *second* association pass against tracks that didn't match a
high-confidence box, using low-confidence boxes only as fallback evidence
for already-established tracks (never to start a brand-new track, since an
unconfirmed low-confidence detection is far more likely to be a false
positive).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from emot.detection.yolo import Detection
from emot.tracking.iou import iou_matrix
from emot.tracking.track import Track, TrackState


@dataclass
class ByteTrackConfig:
    high_conf_threshold: float = 0.6
    low_conf_threshold: float = 0.1
    match_iou_threshold: float = 0.8  # stage 1: IoU distance = 1 - IoU
    low_match_iou_threshold: float = 0.5  # stage 2, looser (recovering occluded boxes)
    min_hits: int = 3
    max_age: int = 30


def _hungarian_iou_match(
    tracks: list[Track], detections: list[Detection], iou_threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    track_boxes = np.array([t.xyxy for t in tracks])
    det_boxes = np.array([d.xyxy for d in detections])
    ious = iou_matrix(track_boxes, det_boxes)
    cost = 1.0 - ious

    row_idx, col_idx = linear_sum_assignment(cost)
    matches, unmatched_tracks, unmatched_dets = [], [], []
    matched_tracks, matched_dets = set(), set()
    for r, c in zip(row_idx, col_idx, strict=True):
        if ious[r, c] >= iou_threshold:
            matches.append((r, c))
            matched_tracks.add(r)
            matched_dets.add(c)
    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


class ByteTracker:
    def __init__(self, config: ByteTrackConfig | None = None):
        self.config = config or ByteTrackConfig()
        self.tracks: list[Track] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        cfg = self.config
        for track in self.tracks:
            track.predict()

        high = [d for d in detections if d.confidence >= cfg.high_conf_threshold]
        low = [
            d
            for d in detections
            if cfg.low_conf_threshold <= d.confidence < cfg.high_conf_threshold
        ]

        # Stage 1: all active tracks vs high-confidence detections.
        matches1, unmatched_track_idx, unmatched_high_idx = _hungarian_iou_match(
            self.tracks, high, 1.0 - cfg.match_iou_threshold
        )
        for t_idx, d_idx in matches1:
            self.tracks[t_idx].update(high[d_idx].xyxy, high[d_idx].confidence)

        # Stage 2: tracks still unmatched vs low-confidence detections
        # (recovers occluded/blurred boxes; never spawns new tracks).
        remaining_tracks = [self.tracks[i] for i in unmatched_track_idx]
        matches2, still_unmatched_idx, _unmatched_low_idx = _hungarian_iou_match(
            remaining_tracks, low, 1.0 - cfg.low_match_iou_threshold
        )
        for t_idx, d_idx in matches2:
            remaining_tracks[t_idx].update(low[d_idx].xyxy, low[d_idx].confidence)

        for i in still_unmatched_idx:
            remaining_tracks[i].mark_missed()

        # New tracks only from *unmatched high-confidence* detections.
        for d_idx in unmatched_high_idx:
            self.tracks.append(Track(high[d_idx].xyxy, high[d_idx].confidence))

        for track in self.tracks:
            track.confirm_if_ready(cfg.min_hits)

        self.tracks = [t for t in self.tracks if t.time_since_update <= cfg.max_age]

        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]
