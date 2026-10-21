"""Frame-in, tracks-out facade over the three tracker families.

The benchmark replays *cached* detections/embeddings/camera-motion; a live
pipeline has only a frame and a detector. This adapter owns the extra
per-frame work each family needs (camera-motion estimation for BoT-SORT,
crop embedding for ReID-based association) so callers just do
`tracker.update(frame, detections)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emot.benchmark.caches import EMBEDDING_MIN_CONF
from emot.benchmark.precompute import crop_rgb
from emot.benchmark.replay import VARIANTS, build_tracker, load_tuned
from emot.detection.yolo import Detection
from emot.reid import Embedder, build_embedder
from emot.tracking.cmc import CameraMotionEstimator


@dataclass(frozen=True)
class TrackedBox:
    track_id: int
    xyxy: np.ndarray
    confidence: float


class OnlineTracker:
    def __init__(self, variant: str = "botsort", device: str = "cpu", tuned: dict | None = None):
        if variant not in VARIANTS:
            raise ValueError(f"unknown tracker {variant!r}; choose from {sorted(VARIANTS)}")
        self.spec = VARIANTS[variant]
        self.tracker = build_tracker(self.spec, load_tuned() if tuned is None else tuned)
        self.embedder: Embedder | None = (
            build_embedder(self.spec.embedder, device) if self.spec.embedder else None
        )
        self.cmc = CameraMotionEstimator() if self.spec.use_cmc else None

    def _embed(self, frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Embeds only detections a tracker can use (>= EMBEDDING_MIN_CONF); the rest stay zero,
        mirroring exactly what the benchmark's embedding cache holds."""
        assert self.embedder is not None
        out = np.zeros((len(detections), self.embedder.dim), dtype=np.float32)
        usable = [i for i, d in enumerate(detections) if d.confidence >= EMBEDDING_MIN_CONF]
        if usable:
            crops = [crop_rgb(frame_bgr, detections[i].xyxy) for i in usable]
            out[usable] = self.embedder.embed_crops(crops)
        return out

    def update(self, frame_bgr: np.ndarray, detections: list[Detection]) -> list[TrackedBox]:
        warp = self.cmc.estimate(frame_bgr) if self.cmc else None
        embeddings = self._embed(frame_bgr, detections) if self.embedder is not None else None

        if self.spec.family == "bytetrack":
            tracks = self.tracker.update(detections)
        elif self.spec.family == "botsort":
            reid = embeddings if self.spec.use_reid else None
            tracks = self.tracker.update(detections, reid, warp)
        else:
            assert embeddings is not None
            tracks = self.tracker.update(detections, embeddings)
        return [TrackedBox(t.track_id, t.xyxy.copy(), float(t.confidence)) for t in tracks]
