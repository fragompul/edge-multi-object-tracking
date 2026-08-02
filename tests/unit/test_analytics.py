import numpy as np

from emot.analytics.trajectories import (
    CountingLine,
    build_trajectories,
    count_line_crossings,
    occupancy_heatmap,
)


def test_build_trajectories_groups_by_track_id_in_frame_order():
    frames = {
        1: {10: np.array([0.0, 0.0, 10.0, 10.0])},
        2: {10: np.array([10.0, 0.0, 20.0, 10.0])},
        3: {10: np.array([20.0, 0.0, 30.0, 10.0]), 11: np.array([0.0, 100.0, 10.0, 110.0])},
    }
    trajectories = build_trajectories(frames)
    assert set(trajectories) == {10, 11}
    assert trajectories[10].frame_indices == [1, 2, 3]
    assert trajectories[10].centers == [(5.0, 5.0), (15.0, 5.0), (25.0, 5.0)]
    assert trajectories[11].frame_indices == [3]


def test_count_line_crossings_detects_a_to_b_and_b_to_a():
    # Vertical line at x=50, from (50,0) to (50,100).
    line = CountingLine(50, 0, 50, 100)
    frames = {
        1: {1: np.array([0.0, 50.0, 10.0, 60.0])},  # left of line (x center=5)
        2: {1: np.array([90.0, 50.0, 100.0, 60.0])},  # right of line (x center=95)
        3: {2: np.array([90.0, 50.0, 100.0, 60.0])},
        4: {2: np.array([0.0, 50.0, 10.0, 60.0])},
    }
    trajectories = build_trajectories(frames)
    counts = count_line_crossings(trajectories, line)
    assert counts["a_to_b"] == 1
    assert counts["b_to_a"] == 1


def test_no_crossing_when_object_stays_on_one_side():
    line = CountingLine(50, 0, 50, 100)
    frames = {
        1: {1: np.array([0.0, 50.0, 10.0, 60.0])},
        2: {1: np.array([5.0, 50.0, 15.0, 60.0])},
        3: {1: np.array([10.0, 50.0, 20.0, 60.0])},
    }
    trajectories = build_trajectories(frames)
    counts = count_line_crossings(trajectories, line)
    assert counts == {"a_to_b": 0, "b_to_a": 0}


def test_occupancy_heatmap_shape_and_normalization():
    frames = {
        i: {1: np.array([float(i * 10), 50.0, float(i * 10 + 10), 60.0])} for i in range(5)
    }
    trajectories = build_trajectories(frames)
    heatmap = occupancy_heatmap(trajectories, frame_width=200, frame_height=200, bins=20)
    assert heatmap.shape == (20, 20)
    assert heatmap.max() == 1.0
    assert heatmap.min() >= 0.0


def test_occupancy_heatmap_empty_trajectories_returns_zeros():
    heatmap = occupancy_heatmap({}, frame_width=200, frame_height=200, bins=10)
    assert heatmap.shape == (10, 10)
    assert np.all(heatmap == 0)
