import numpy as np
import pytest

from emot.analytics.events import (
    Zone,
    crossing_events,
    detect_crowding,
    detect_loitering,
    dwell_stats,
    motion_stats,
    point_in_polygon,
    scene_summary,
)
from emot.analytics.trajectories import CountingLine, build_trajectories

SQUARE = Zone("plaza", ((0, 0), (100, 0), (100, 100), (0, 100)))


def _box_at(foot_x: float, foot_y: float, h: float = 50.0) -> np.ndarray:
    return np.array([foot_x - 10, foot_y - h, foot_x + 10, foot_y])


def test_point_in_polygon_handles_concave_shapes():
    ell = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    assert point_in_polygon((2, 2), ell)
    assert point_in_polygon((2, 8), ell)
    assert not point_in_polygon((8, 8), ell)  # the notch of the L


def test_dwell_uses_foot_point_and_counts_entries():
    # inside for 4 frames, out for 2, back in for 2 (30 fps)
    xs = [50, 50, 50, 50, 500, 500, 50, 50]
    frames = {i + 1: {7: _box_at(x, 60)} for i, x in enumerate(xs)}
    stats = dwell_stats(build_trajectories(frames), SQUARE, fps=30)
    assert stats[7]["seconds"] == pytest.approx(6 / 30)
    assert stats[7]["entries"] == 2
    assert stats[7]["first_frame"] == 1


def test_loitering_fires_once_at_threshold_crossing():
    frames = {i: {1: _box_at(50, 60)} for i in range(1, 91)}  # 3 s standing still at 30 fps
    events = detect_loitering(build_trajectories(frames), SQUARE, fps=30, min_seconds=2.0)
    assert len(events) == 1 and events[0].frame == 60 and events[0].track_id == 1
    assert detect_loitering(build_trajectories(frames), SQUARE, 30, min_seconds=5.0) == []


def test_crowding_reports_stretches_with_peak():
    def frame(n):
        return {k: _box_at(10 + 5 * k, 60) for k in range(n)}

    counts = [1, 1, 4, 5, 4, 1, 1, 4, 1]  # one 3-frame stretch >=4, one 1-frame blip
    frames = {i + 1: frame(n) for i, n in enumerate(counts)}
    events = detect_crowding(frames, SQUARE, fps=3, threshold=4, min_seconds=1.0)
    assert len(events) == 1
    assert events[0].frame == 3 and events[0].details["end_frame"] == 5
    assert events[0].details["peak"] == 5


def test_crossing_events_carry_direction_and_frame():
    line = CountingLine(50, 0, 50, 100)
    frames = {1: {1: _box_at(10, 50)}, 2: {1: _box_at(90, 50)}, 3: {1: _box_at(10, 50)}}
    events = crossing_events(build_trajectories(frames), line, fps=1)
    # A = side where cross((p2-p1), (p-p1)) < 0 (here: right of the downward line)
    assert [e.details["direction"] for e in events] == ["b_to_a", "a_to_b"]
    assert [e.frame for e in events] == [2, 3]


def test_motion_stats_are_perspective_invariant():
    """The same walk at two scales (near/far from camera) gives the same speed in body-heights/s."""

    def walk(scale):
        return {
            i: {1: np.array([100 + 5 * scale * i, 100, 120 + 5 * scale * i, 100 + 50 * scale])}
            for i in range(1, 31)
        }

    near = motion_stats(build_trajectories(walk(3.0)), fps=30)[1]["median_speed_bh_per_s"]
    far = motion_stats(build_trajectories(walk(1.0)), fps=30)[1]["median_speed_bh_per_s"]
    assert near == pytest.approx(far, rel=1e-6)


def test_scene_summary_reports_peak():
    frames = {1: {1: _box_at(10, 50)}, 2: {1: _box_at(10, 50), 2: _box_at(60, 50)}}
    summary = scene_summary(frames, build_trajectories(frames), fps=2)
    assert summary["unique_tracks"] == 2 and summary["peak_people"] == 2
    assert summary["peak_frame"] == 2 and summary["duration_s"] == 1.0
