import numpy as np

from emot.detection.yolo import Detection
from emot.tracking.deepsort import DeepSortConfig, DeepSortTracker


def _random_unit_embedding(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(512)
    return v / np.linalg.norm(v)


def _det(x: float) -> Detection:
    return Detection(xyxy=np.array([x, 100.0, x + 50.0, 200.0]), confidence=0.9)


def test_consistent_appearance_keeps_same_track_id():
    rng = np.random.default_rng(0)
    base = _random_unit_embedding(rng)
    tracker = DeepSortTracker(DeepSortConfig(min_hits=2))

    ids_over_time = []
    for t in range(6):
        embedding = base + rng.standard_normal(512) * 0.01
        embedding /= np.linalg.norm(embedding)
        tracks = tracker.update([_det(100 + t * 10)], np.array([embedding]))
        ids_over_time.append({tr.track_id for tr in tracks})

    confirmed = [ids for ids in ids_over_time if ids]
    assert confirmed
    first_id = next(iter(confirmed[0]))
    assert all(first_id in ids for ids in confirmed)


def test_two_objects_with_distinct_appearance_get_different_ids():
    rng = np.random.default_rng(1)
    emb_a = _random_unit_embedding(rng)
    emb_b = _random_unit_embedding(rng)
    tracker = DeepSortTracker(DeepSortConfig(min_hits=2))

    for t in range(4):
        dets = [_det(100 + t * 5), _det(500 + t * 5)]
        embeddings = np.array([emb_a, emb_b])
        tracks = tracker.update(dets, embeddings)

    assert len({tr.track_id for tr in tracks}) == 2


def test_empty_detections_do_not_crash_and_return_no_tracks():
    tracker = DeepSortTracker()
    tracks = tracker.update([], np.zeros((0, 512)))
    assert tracks == []


def test_track_dropped_after_max_age():
    tracker = DeepSortTracker(DeepSortConfig(min_hits=1, max_age=2))
    rng = np.random.default_rng(2)
    emb = _random_unit_embedding(rng)
    tracker.update([_det(100)], np.array([emb]))
    assert len(tracker.tracks) == 1
    for _ in range(4):
        tracker.update([], np.zeros((0, 512)))
    assert len(tracker.tracks) == 0
