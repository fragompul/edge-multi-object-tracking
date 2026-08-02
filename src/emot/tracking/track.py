"""Shared per-track state used by both ByteTrack and DeepSORT.

A track's lifecycle: `tentative` (seen too few times to trust yet) ->
`confirmed` (emitted in output) -> `lost` (missed >= 1 frame, kept alive
in case it reappears) -> removed once missed for too long.
"""

from __future__ import annotations

import itertools
from enum import Enum, auto

import numpy as np

from emot.tracking.kalman import KalmanBoxTracker

_id_counter = itertools.count(1)


class TrackState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    LOST = auto()


class Track:
    def __init__(self, xyxy: np.ndarray, confidence: float, embedding: np.ndarray | None = None):
        self.track_id = next(_id_counter)
        self.kalman = KalmanBoxTracker(xyxy)
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.time_since_update = 0
        self.confidence = confidence
        # A short rolling gallery of recent appearance embeddings (DeepSORT
        # only) -- matching against several recent looks, not just the
        # last one, is more robust to a single bad-angle frame.
        self.embeddings: list[np.ndarray] = [embedding] if embedding is not None else []

    @property
    def xyxy(self) -> np.ndarray:
        return self.kalman.xyxy

    def predict(self) -> None:
        self.kalman.predict()
        self.time_since_update += 1

    def update(
        self, xyxy: np.ndarray, confidence: float, embedding: np.ndarray | None = None
    ) -> None:
        self.kalman.update(xyxy)
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        if embedding is not None:
            self.embeddings.append(embedding)
            if len(self.embeddings) > 10:
                self.embeddings.pop(0)

    def mark_missed(self) -> None:
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.LOST  # dropped next cleanup pass
        elif self.time_since_update > 0:
            self.state = TrackState.LOST

    def confirm_if_ready(self, min_hits: int) -> None:
        ready_from_tentative = self.state == TrackState.TENTATIVE and self.hits >= min_hits
        ready_from_lost = self.state == TrackState.LOST and self.time_since_update == 0
        if ready_from_tentative or ready_from_lost:
            self.state = TrackState.CONFIRMED
