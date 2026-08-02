"""Derives per-track trajectories, line-crossing counts, and a spatial
occupancy heatmap from a sequence's frame-by-frame tracker output.

All three are computed from the same underlying data (`{frame_idx:
{track_id: xyxy}}`), kept as one pass over the recorded track history
rather than three separate re-scans, since a full sequence can have
hundreds of frames x tens of tracks.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

FrameTracks = dict[int, np.ndarray]  # track_id -> xyxy


@dataclass
class TrackHistory:
    frame_indices: list[int] = field(default_factory=list)
    centers: list[tuple[float, float]] = field(default_factory=list)

    def as_array(self) -> np.ndarray:
        return np.array(self.centers)


def build_trajectories(frames: dict[int, FrameTracks]) -> dict[int, TrackHistory]:
    trajectories: dict[int, TrackHistory] = defaultdict(TrackHistory)
    for frame_idx in sorted(frames):
        for track_id, xyxy in frames[frame_idx].items():
            cx = float((xyxy[0] + xyxy[2]) / 2)
            cy = float((xyxy[1] + xyxy[3]) / 2)
            trajectories[track_id].frame_indices.append(frame_idx)
            trajectories[track_id].centers.append((cx, cy))
    return dict(trajectories)


@dataclass(frozen=True)
class CountingLine:
    """A directed line segment `(x1, y1) -> (x2, y2)`; a track crossing it
    increments the count for whichever side it moved *into* (sign of the
    2D cross product between the line direction and the track's motion
    vector at the crossing)."""

    x1: float
    y1: float
    x2: float
    y2: float


def _side(line: CountingLine, point: tuple[float, float]) -> float:
    return (line.x2 - line.x1) * (point[1] - line.y1) - (line.y2 - line.y1) * (
        point[0] - line.x1
    )


def count_line_crossings(
    trajectories: dict[int, TrackHistory], line: CountingLine
) -> dict[str, int]:
    """Returns `{"a_to_b": n, "b_to_a": n}` -- a track counts once per
    crossing (a consecutive pair of points that straddle the line), so a
    track that lingers near the line and jitters across it repeatedly is
    counted each time, which is the same "did a person actually cross"
    semantics a real people-counter needs (someone pacing back and forth
    at a doorway really did cross multiple times)."""
    counts = {"a_to_b": 0, "b_to_a": 0}
    for history in trajectories.values():
        centers = history.centers
        for i in range(1, len(centers)):
            prev_side = _side(line, centers[i - 1])
            curr_side = _side(line, centers[i])
            if prev_side == 0 or curr_side == 0:
                continue
            if (prev_side > 0) != (curr_side > 0):
                if prev_side < 0:
                    counts["a_to_b"] += 1
                else:
                    counts["b_to_a"] += 1
    return counts


def occupancy_heatmap(
    trajectories: dict[int, TrackHistory], frame_width: int, frame_height: int, bins: int = 50
) -> np.ndarray:
    """2D histogram of track-center positions over the whole sequence,
    normalized to `[0, 1]` -- the standard "where did people spend time"
    view."""
    non_empty = [history.as_array() for history in trajectories.values() if history.centers]
    if not non_empty:
        return np.zeros((bins, bins))
    all_points = np.concatenate(non_empty)

    heatmap, _, _ = np.histogram2d(
        all_points[:, 0],
        all_points[:, 1],
        bins=bins,
        range=[[0, frame_width], [0, frame_height]],
    )
    max_val = heatmap.max()
    return heatmap / max_val if max_val > 0 else heatmap
