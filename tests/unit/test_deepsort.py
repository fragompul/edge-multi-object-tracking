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


def test_vectorised_mahalanobis_matches_pairwise_gate():
    from emot.tracking.kalman import KalmanBoxTracker, mahalanobis_matrix

    rng = np.random.default_rng(1)
    trackers = []
    for k in range(4):
        kf = KalmanBoxTracker(np.array([50.0 + 90 * k, 80.0, 90.0 + 90 * k, 200.0]))
        kf.predict()
        trackers.append(kf)
    boxes = np.array([[40, 70, 85, 195], [200, 90, 250, 210], [5, 5, 30, 90]], dtype=float)
    boxes = boxes + rng.normal(0, 2, boxes.shape)
    fast = mahalanobis_matrix(trackers, boxes)
    slow = np.array([[kf.mahalanobis_gate(b) for b in boxes] for kf in trackers])
    assert np.allclose(fast, slow)


def test_cascade_prefers_recently_seen_track_for_contested_detection():
    """Two tracks both plausibly match one detection; the one seen last frame
    must win under the matching cascade even if the stale one looks more similar."""
    from emot.tracking.track import Track, TrackState

    config = DeepSortConfig(cascade=True, max_cosine_distance=0.5, min_hits=1)
    tracker = DeepSortTracker(config)
    emb_fresh = np.array([1.0, 0.0, 0.0])
    emb_stale = np.array([0.9, 0.436, 0.0])
    emb_stale /= np.linalg.norm(emb_stale)
    det_emb = np.array([[0.95, 0.31, 0.0]])
    det_emb /= np.linalg.norm(det_emb)

    fresh = Track(np.array([100.0, 100.0, 140.0, 200.0]), 0.9, emb_fresh)
    stale = Track(np.array([102.0, 100.0, 142.0, 200.0]), 0.9, emb_stale)
    for trk in (fresh, stale):
        trk.state = TrackState.CONFIRMED
    stale.time_since_update = 4
    tracker.tracks = [fresh, stale]
    det = Detection(xyxy=np.array([101.0, 100.0, 141.0, 200.0]), confidence=0.9)

    matches, _, _ = tracker._match([det], det_emb)
    assert matches == [(0, 0)]
