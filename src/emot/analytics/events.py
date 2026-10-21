"""Scene-level analytics on top of tracks: zones, dwell time, loitering,
crowding, line-crossing events, and scale-free motion statistics.

All geometry works on the **foot point** (bottom-centre of the box). Pixel
speeds are not comparable across a perspective scene (a person near the
camera covers more pixels per step than a distant one walking just as
fast), so speed is reported in **body-heights per second** -- a unit that
cancels the perspective scale without needing camera calibration.

Privacy note: everything here consumes anonymous, session-local track IDs
and box geometry. No appearance data is needed or retained.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from emot.analytics.trajectories import CountingLine, FrameTracks, TrackHistory, _side

Point = tuple[float, float]


@dataclass(frozen=True)
class Zone:
    """A named polygon in pixel coordinates."""

    name: str
    polygon: tuple[Point, ...]

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)

    @staticmethod
    def from_normalized(name: str, points: Sequence[Point], width: int, height: int) -> Zone:
        return Zone(name, tuple((x * width, y * height) for x, y in points))


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Even-odd ray casting; points exactly on an edge count as outside-or-inside
    arbitrarily, which is irrelevant at pixel scale."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


@dataclass(frozen=True)
class Event:
    kind: str  # "crossing" | "loitering" | "crowding"
    frame: int
    seconds: float
    track_id: int | None = None
    zone: str | None = None
    details: dict = field(default_factory=dict)


def foot_point(xyxy: np.ndarray) -> Point:
    return (float((xyxy[0] + xyxy[2]) / 2), float(xyxy[3]))


def dwell_stats(
    trajectories: dict[int, TrackHistory], zone: Zone, fps: float
) -> dict[int, dict[str, float]]:
    """Per-track seconds spent inside `zone`, number of entries, and first frame inside."""
    out: dict[int, dict[str, float]] = {}
    for tid, hist in trajectories.items():
        inside = [zone.contains(p) for p in hist.foot_points]
        frames_in = int(sum(inside))
        if frames_in == 0:
            continue
        entries = int(inside[0]) + sum(
            1 for a, b in zip(inside, inside[1:], strict=False) if b and not a
        )
        first = next(f for f, flag in zip(hist.frame_indices, inside, strict=True) if flag)
        out[tid] = {
            "seconds": frames_in / fps,
            "entries": entries,
            "first_frame": first,
        }
    return out


def detect_loitering(
    trajectories: dict[int, TrackHistory], zone: Zone, fps: float, min_seconds: float
) -> list[Event]:
    """Tracks that spend at least `min_seconds` inside the zone (reported once, at the
    frame where the threshold was first exceeded)."""
    events = []
    need = int(np.ceil(min_seconds * fps))
    for tid, hist in trajectories.items():
        count = 0
        for frame, p in zip(hist.frame_indices, hist.foot_points, strict=True):
            if zone.contains(p):
                count += 1
                if count == need:
                    events.append(
                        Event(
                            "loitering",
                            frame,
                            frame / fps,
                            tid,
                            zone.name,
                            {"min_seconds": min_seconds},
                        )
                    )
                    break
    return sorted(events, key=lambda e: e.frame)


def occupancy_series(frames: dict[int, FrameTracks], zone: Zone | None) -> dict[int, int]:
    """People count per frame (inside `zone`, or everywhere if `zone` is None)."""
    series = {}
    for frame in sorted(frames):
        boxes = frames[frame].values()
        series[frame] = sum(1 for b in boxes if zone is None or zone.contains(foot_point(b)))
    return series


def detect_crowding(
    frames: dict[int, FrameTracks],
    zone: Zone | None,
    fps: float,
    threshold: int,
    min_seconds: float = 1.0,
) -> list[Event]:
    """Contiguous stretches where occupancy stays >= `threshold` for `min_seconds`."""
    series = occupancy_series(frames, zone)
    need = max(1, int(round(min_seconds * fps)))
    zone_name = zone.name if zone else None
    events: list[Event] = []
    run: list[tuple[int, int]] = []  # (frame, occupancy) of the current stretch

    def close_run() -> None:
        if len(run) >= need:
            events.append(
                Event(
                    "crowding",
                    run[0][0],
                    run[0][0] / fps,
                    None,
                    zone_name,
                    {
                        "end_frame": run[-1][0],
                        "peak": max(n for _, n in run),
                        "threshold": threshold,
                    },
                )
            )
        run.clear()

    prev_frame = None
    for frame, n in series.items():
        if n >= threshold and (prev_frame is None or not run or frame == prev_frame + 1):
            run.append((frame, n))
        else:
            close_run()
            if n >= threshold:
                run.append((frame, n))
        prev_frame = frame
    close_run()
    return events


def crossing_events(
    trajectories: dict[int, TrackHistory], line: CountingLine, fps: float, name: str = "line"
) -> list[Event]:
    """Every crossing as an event (frame, track, direction); counts are `len()` of the
    events per direction. Uses foot points, like the zones.

    Direction is relative to the line's orientation: side **A** is where the 2D cross
    product `(p2 - p1) x (p - p1)` is negative, side **B** where it is positive."""
    events = []
    for tid, hist in trajectories.items():
        pts = hist.foot_points
        for i in range(1, len(pts)):
            a, b = _side(line, pts[i - 1]), _side(line, pts[i])
            if a == 0 or b == 0 or (a > 0) == (b > 0):
                continue
            direction = "a_to_b" if a < 0 else "b_to_a"
            frame = hist.frame_indices[i]
            details = {"direction": direction}
            events.append(Event("crossing", frame, frame / fps, tid, name, details))
    return sorted(events, key=lambda e: e.frame)


def motion_stats(
    trajectories: dict[int, TrackHistory], fps: float, min_frames: int = 10
) -> dict[int, dict[str, float]]:
    """Median speed in body-heights/second and total path length, per track."""
    out = {}
    for tid, hist in trajectories.items():
        if len(hist.foot_points) < min_frames:
            continue
        pts = np.array(hist.foot_points)
        frames = np.array(hist.frame_indices, dtype=float)
        dt = np.diff(frames) / fps
        keep = dt > 0
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)[keep]
        height = np.array(hist.heights)[1:][keep]
        speed = step / np.maximum(height, 1.0) / dt[keep]
        out[tid] = {
            "median_speed_bh_per_s": float(np.median(speed)),
            "path_px": float(step.sum()),
            "duration_s": float((frames[-1] - frames[0]) / fps),
        }
    return out


def scene_summary(
    frames: dict[int, FrameTracks], trajectories: dict[int, TrackHistory], fps: float
) -> dict:
    series = occupancy_series(frames, None)
    peak_frame = max(series, key=lambda f: series[f]) if series else 0
    durations = [
        (h.frame_indices[-1] - h.frame_indices[0] + 1) / fps for h in trajectories.values()
    ]
    return {
        "frames": len(frames),
        "duration_s": round(len(frames) / fps, 2),
        "unique_tracks": len(trajectories),
        "mean_people_per_frame": round(float(np.mean(list(series.values()))) if series else 0.0, 2),
        "peak_people": series.get(peak_frame, 0),
        "peak_frame": peak_frame,
        "peak_time_s": round(peak_frame / fps, 2),
        "median_track_duration_s": round(float(np.median(durations)), 2) if durations else 0.0,
    }
