import numpy as np

from emot.evaluation.mot_metrics import MOTAccumulator
from emot.tracking.track import Track, TrackState


def _confirmed_track(xyxy: np.ndarray) -> Track:
    track = Track(xyxy, confidence=0.9)
    track.state = TrackState.CONFIRMED
    return track


def test_perfect_tracking_gives_mota_and_idf1_of_one():
    accumulator = MOTAccumulator()
    track = _confirmed_track(np.array([100.0, 100.0, 150.0, 200.0]))
    for t in range(5):
        gt = np.array([[track.track_id, 100 + t * 5, 100, 50, 100]], dtype=float)
        # Move the (fixed-id) track to exactly match ground truth each frame.
        track.kalman.mean[:4] = np.array(
            [125 + t * 5, 150, 0.5, 100]
        )  # cx, cy, aspect, height matching the gt box
        accumulator.update(gt, [track])
    metrics = accumulator.compute("perfect")
    assert metrics.mota == 1.0
    assert metrics.idf1 == 1.0
    assert metrics.num_switches == 0


def test_no_detections_gives_all_misses():
    accumulator = MOTAccumulator()
    for _ in range(3):
        gt = np.array([[1, 100, 100, 50, 100]], dtype=float)
        accumulator.update(gt, [])
    metrics = accumulator.compute("misses")
    assert metrics.num_misses == 3
    # MOTA = 1 - (FN + FP + IDSW) / num_gt = 1 - 3/3 = 0 (the CLEAR-MOT floor
    # for "detected nothing," not a negative score).
    assert metrics.mota == 0.0


def test_false_positive_only_no_gt():
    accumulator = MOTAccumulator()
    track = _confirmed_track(np.array([500.0, 500.0, 550.0, 600.0]))
    for _ in range(3):
        accumulator.update(np.zeros((0, 5)), [track])
    metrics = accumulator.compute("fp-only")
    assert metrics.num_false_positives == 3


def test_identity_switch_detected_when_track_id_changes():
    accumulator = MOTAccumulator()
    gt = np.array([[1, 100, 100, 50, 100]], dtype=float)
    accumulator.update(gt, [_confirmed_track(np.array([100.0, 100.0, 150.0, 200.0]))])
    accumulator.update(gt, [_confirmed_track(np.array([100.0, 100.0, 150.0, 200.0]))])
    metrics = accumulator.compute("switch")
    assert metrics.num_switches == 1
