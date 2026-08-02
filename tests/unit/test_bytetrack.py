import numpy as np

from emot.detection.yolo import Detection
from emot.tracking.bytetrack import ByteTrackConfig, ByteTracker
from emot.tracking.track import TrackState


def _det(x: float, confidence: float = 0.9) -> Detection:
    return Detection(xyxy=np.array([x, 100.0, x + 50.0, 200.0]), confidence=confidence)


def test_single_object_gets_one_persistent_track_id():
    tracker = ByteTracker(ByteTrackConfig(min_hits=2))
    ids_over_time = []
    for t in range(6):
        tracks = tracker.update([_det(100 + t * 10)])
        ids_over_time.append({tr.track_id for tr in tracks})

    confirmed_ids = [ids for ids in ids_over_time if ids]
    assert len(confirmed_ids) > 0
    # Once confirmed, the same track_id persists across every later frame.
    first_confirmed_id = next(iter(confirmed_ids[0]))
    assert all(first_confirmed_id in ids for ids in confirmed_ids)


def test_low_confidence_detection_never_starts_a_new_track():
    tracker = ByteTracker(ByteTrackConfig(high_conf_threshold=0.6, low_conf_threshold=0.1))
    tracks = tracker.update([_det(100, confidence=0.3)])
    assert tracks == []
    assert tracker.tracks == []  # not even a tentative track


def test_two_well_separated_objects_get_different_ids():
    tracker = ByteTracker(ByteTrackConfig(min_hits=2))
    for t in range(4):
        tracks = tracker.update([_det(100 + t * 5), _det(500 + t * 5)])
    assert len({tr.track_id for tr in tracks}) == 2


def test_track_is_dropped_after_max_age_missed_frames():
    tracker = ByteTracker(ByteTrackConfig(min_hits=1, max_age=3))
    tracker.update([_det(100)])
    tracker.update([_det(110)])
    assert len(tracker.tracks) == 1
    for _ in range(5):
        tracker.update([])  # object disappears
    assert len(tracker.tracks) == 0


def test_confirmed_tracks_never_returned_before_min_hits():
    tracker = ByteTracker(ByteTrackConfig(min_hits=5))
    for t in range(3):
        tracks = tracker.update([_det(100 + t * 5)])
        assert tracks == []
    assert tracker.tracks[0].state != TrackState.CONFIRMED
